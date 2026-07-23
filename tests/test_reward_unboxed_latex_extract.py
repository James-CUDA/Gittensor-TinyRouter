"""Un-boxed LaTeX \\frac/\\sqrt answers must be extracted whole (issue #419)."""
from __future__ import annotations

from trinity.orchestration.reward import extract_last_number, score_text


def test_frac_not_read_as_denominator():
    assert extract_last_number(r"So the answer is $\frac{1}{2}$.") == r"\frac{1}{2}"
    assert score_text("math500", r"So the answer is $\frac{1}{2}$.", r"\frac{1}{2}") == 1.0
    assert score_text("math500", r"So the answer is $\frac{1}{2}$.", "2") == 0.0


def test_coeff_sqrt_kept_whole():
    assert extract_last_number(r"Therefore the area is $2\sqrt{3}$.") == r"2\sqrt{3}"


def test_later_plain_number_still_wins():
    assert extract_last_number(r"$\frac{1}{2}$ so approximately 0.5") == "0.5"
