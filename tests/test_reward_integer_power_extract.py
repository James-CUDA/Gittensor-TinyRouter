"""Unboxed ``10^{3}`` must not grade as the exponent digit (issue #499)."""

from trinity.orchestration.reward import extract_last_number, score_text


def test_extract_power_as_whole_term():
    # spaces stripped inside extract candidates
    assert extract_last_number(r"10^{3}") in (r"10^{3}", "10^{3}")
    assert extract_last_number(r"$10^{3}$") in (r"10^{3}", "10^{3}")
    assert extract_last_number(r"10^3") == "10^3"


def test_unboxed_power_grades_value_not_exponent():
    assert score_text("math500", r"the answer is $10^{3}$", "1000") == 1.0
    assert score_text("math500", r"the answer is $10^{3}$", "3") == 0.0


def test_boxed_power_still_works():
    assert score_text("math500", r"\boxed{10^{3}}", "1000") == 1.0
    assert score_text("math500", r"\boxed{2^{10}}", "1024") == 1.0
