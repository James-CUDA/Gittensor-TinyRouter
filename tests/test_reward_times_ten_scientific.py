"""Unboxed ``a\\times 10^{b}`` must not grade as the exponent digit (issue #485)."""

from trinity.orchestration.reward import extract_last_number, score_text


def test_extract_times_ten_as_whole_term():
    # Spaces inside the match are stripped (same as other extract candidates).
    assert extract_last_number(r"3\times 10^{2}") == r"3\times10^{2}"
    assert extract_last_number(r"answer is $3\times 10^{2}$") == r"3\times10^{2}"
    assert extract_last_number(r"2.5\times 10^3") == r"2.5\times10^3"


def test_unboxed_times_ten_grades_value_not_exponent():
    assert score_text("math500", r"the answer is $3\times 10^{2}$", "300") == 1.0
    assert score_text("math500", r"the answer is $3\times 10^{2}$", "2") == 0.0


def test_boxed_times_ten_still_works():
    assert score_text("math500", r"\boxed{3\times 10^{2}}", "300") == 1.0


def test_ascii_scientific_still_works():
    assert extract_last_number("the answer is 1e3") == "1e3"
    assert score_text("math500", "the answer is 1e3", "1000") == 1.0
