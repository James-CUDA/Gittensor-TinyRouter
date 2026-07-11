"""LaTeX font/style commands are unwrapped to their content in the math scorer.

``normalize_math_answer`` already unwraps ``\\text{...}`` and ``\\mathrm{...}``.
These commands only change how an answer *looks*, not its value, so the other
common font wrappers (``\\mathbf``, ``\\boldsymbol``, ``\\mathit``, ...) must be
unwrapped too — otherwise ``\\boxed{\\mathbf{5}}`` scores 0 against a plain ``5``.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

import pytest

from trinity.orchestration.reward import normalize_math_answer, score_text


@pytest.mark.parametrize(
    "wrapped,inner",
    [
        (r"\mathbf{5}", "5"),
        (r"\boldsymbol{7}", "7"),
        (r"\mathit{3}", "3"),
        (r"\mathsf{42}", "42"),
        (r"\mathtt{9}", "9"),
        (r"\text{5}", "5"),      # regression: still unwrapped
        (r"\mathrm{5}", "5"),    # regression: still unwrapped
    ],
)
def test_font_commands_unwrap_to_content(wrapped, inner):
    assert normalize_math_answer(wrapped) == inner


def test_bold_answer_scores_correct():
    assert score_text("math500", r"The answer is \boxed{\mathbf{5}}.", "5") == 1.0
    assert score_text("math500", r"\boxed{\boldsymbol{42}}", "42") == 1.0
    # A wrong bold value is still wrong.
    assert score_text("math500", r"\boxed{\mathbf{6}}", "5") == 0.0


def test_bold_matches_plain_reference_and_vice_versa():
    assert score_text("math500", r"\boxed{\mathbf{5}}", "5") == 1.0
    assert score_text("math500", r"\boxed{5}", r"\mathbf{5}") == 1.0


def test_frac_is_not_treated_as_a_font_command():
    # \frac must be handled by the fraction rule, not swallowed by the unwrap.
    assert normalize_math_answer(r"\frac{1}{2}") == "1/2"


# --------------------------------------------------------------------------- #
# Font wrappers around a BRACED payload (fraction / root / nested font). The
# original single-level ``[^{}]*`` regex could not match these, so a bold
# fraction scored 0 against a plain reference.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "wrapped,normalized",
    [
        (r"\mathbf{\frac{1}{2}}", "1/2"),
        (r"\boldsymbol{\frac{3}{4}}", "3/4"),
        (r"\mathbf{\sqrt{2}}", r"\sqrt{2}"),
        (r"\mathrm{\frac{5}{6}}", "5/6"),
        (r"\mathbf{\mathrm{5}}", "5"),          # nested font commands
        (r"\text{\mathbf{7}}", "7"),
    ],
)
def test_font_command_around_braced_payload_unwraps(wrapped, normalized):
    assert normalize_math_answer(wrapped) == normalize_math_answer(normalized)


def test_bold_fraction_scores_correct_against_plain_reference():
    assert score_text("math500", r"\boxed{\mathbf{\frac{1}{2}}}", "1/2") == 1.0
    assert score_text("math500", r"The answer is \boxed{\boldsymbol{\frac{3}{4}}}.", "3/4") == 1.0
    assert score_text("math500", r"\boxed{\mathbf{\sqrt{2}}}", r"\sqrt{2}") == 1.0
    # A wrong bold fraction is still wrong.
    assert score_text("math500", r"\boxed{\mathbf{\frac{1}{3}}}", "1/2") == 0.0


def test_unbalanced_font_wrapper_does_not_crash_or_corrupt():
    # Malformed input must be left intact (no exception, no truncation past the
    # missing brace beyond the normal single-outer-brace handling).
    out = normalize_math_answer(r"\mathbf{\frac{1}{2}")
    assert isinstance(out, str)
