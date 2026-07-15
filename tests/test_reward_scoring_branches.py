"""Branch coverage for the core scorer `trinity.orchestration.reward`.

The common scoring paths (`score_text`, `normalize_math_answer`, `math_equal`,
`has_answer`) are heavily exercised elsewhere, but many **edge branches** of the
scoring core were uncovered (reward.py at 84%): boxed/number extraction guards,
LaTeX normalization corner cases, choice-reference coercion, code test-spec
coercion, the three code-test execution flavours, and `_terminating_role`.

These are the correctness surface of grading — a broken branch here silently
mis-scores a benchmark. All tests are offline (the code-execution tests spawn a
short-lived `sys.executable` subprocess exactly as production does). No torch.
"""
from __future__ import annotations

import sys

from trinity.orchestration import reward as R
from trinity.types import Role, Task, Trajectory, TurnRecord


def test_no_torch_imported():
    assert "torch" not in sys.modules, "reward scoring tests must not import torch"


# --------------------------------------------------------------------------- #
# extract_boxed / extract_last_number
# --------------------------------------------------------------------------- #
def test_extract_boxed_skips_whitespace_before_brace():
    assert R.extract_boxed(r"\boxed {5}") == "5"


def test_extract_boxed_skips_occurrence_without_a_brace():
    # First "\boxed x" has no brace -> scan continues to the real boxed answer.
    assert R.extract_boxed(r"a \boxed x then \boxed{9}") == "9"


def test_extract_boxed_unbalanced_returns_none():
    assert R.extract_boxed(r"\boxed{5") is None


def test_extract_last_number_empty_is_none():
    assert R.extract_last_number("") is None


# --------------------------------------------------------------------------- #
# normalize_math_answer / _unwrap_font_commands / _as_number
# --------------------------------------------------------------------------- #
def test_normalize_none_is_empty_string():
    assert R.normalize_math_answer(None) == ""


def test_normalize_strips_leading_equals():
    assert R.normalize_math_answer("= 5") == "5"


def test_normalize_zero_denominator_fraction_is_left_as_text():
    # Fraction(3, 0) raises -> the canonicalization is skipped, string kept.
    assert R.normalize_math_answer("3/0") == "3/0"


def test_unwrap_font_command_without_brace_is_unchanged():
    assert R._unwrap_font_commands(r"\mathbf 5") == r"\mathbf 5"


def test_as_number_empty_is_none():
    assert R._as_number("") is None


def test_as_number_parses_and_rejects_fractions():
    assert R._as_number("3/4") == 0.75
    assert R._as_number("5/0") is None       # zero denominator -> None
    assert R._as_number("abc") is None


def test_sympy_equal_empty_operand_is_false():
    assert R._sympy_equal("", "5") is False


# --------------------------------------------------------------------------- #
# _check_math / _ref_to_str via score_text
# --------------------------------------------------------------------------- #
def test_score_text_math_reference_may_be_boxed():
    assert R.score_text("math500", r"\boxed{5}", r"\boxed{5}") == 1.0


def test_score_text_math_accepts_non_string_reference():
    # int reference -> _ref_to_str -> "5"; matches boxed 5.
    assert R.score_text("math500", r"\boxed{5}", 5) == 1.0


def test_ref_to_str_none_is_empty():
    assert R._ref_to_str(None) == ""
    assert R._ref_to_str(7) == "7"


# --------------------------------------------------------------------------- #
# choice extraction / reference coercion
# --------------------------------------------------------------------------- #
def test_extract_choice_letter_empty_is_none():
    assert R.extract_choice_letter("") is None


def test_choice_reference_none_scores_zero():
    assert R.score_text("mmlu", "Answer: B", None) == 0.0


def test_normalize_reference_letter_variants():
    assert R.normalize_reference_letter(None) is None
    assert R.normalize_reference_letter("Z") is None      # not an A-J letter
    assert R.normalize_reference_letter(True) is None      # bool is not an index
    assert R.normalize_reference_letter(99) is None        # index out of range
    assert R.normalize_reference_letter(2.5) is None       # non-int/str
    assert R.normalize_reference_letter(2) == "C"          # 0-based index


# --------------------------------------------------------------------------- #
# code: extract_code / _coerce_test_spec / _check_code
# --------------------------------------------------------------------------- #
def test_extract_code_empty_is_empty_string():
    assert R.extract_code("") == ""


def test_coerce_test_spec_invalid_json_string_becomes_single_assert():
    tests, timeout_s, fn = R._coerce_test_spec("assert f() == 1")
    assert tests == ["assert f() == 1"]
    assert timeout_s == 10 and fn is None


def test_coerce_test_spec_dict_none_tests_and_non_list_tests():
    assert R._coerce_test_spec({"tests": None}) == ([], 10, None)
    assert R._coerce_test_spec({"tests": "one"}) == (["one"], 10, None)


def test_coerce_test_spec_reads_timeout_and_fn_name():
    tests, timeout_s, fn = R._coerce_test_spec(
        {"tests": ["assert True"], "timeout_s": 3, "fn_name": "solve"}
    )
    assert tests == ["assert True"] and timeout_s == 3 and fn == "solve"


def test_check_code_empty_candidate_is_false():
    assert R._check_code("   ", ["assert True"]) is False


# --------------------------------------------------------------------------- #
# code execution: run_pass_at_1 flavours (real subprocess)
# --------------------------------------------------------------------------- #
def test_run_pass_at_1_empty_tests_is_false():
    assert R.run_pass_at_1("x = 1", []) is False


def test_run_pass_at_1_empty_code_is_false():
    assert R.run_pass_at_1("   ", ["assert True"]) is False


def test_run_pass_at_1_assert_key_dict():
    assert R.run_pass_at_1("z = 7", [{"assert": "assert z == 7"}]) is True


def test_run_pass_at_1_stdin_test_script_error_is_false():
    # The candidate crashes (non-zero exit) -> stdin/stdout test fails on `not ok`.
    assert R.run_pass_at_1("raise SystemExit(1)",
                           [{"stdin": "", "expected_stdout": "x"}]) is False


def test_run_pass_at_1_timeout_is_false():
    # A candidate that never terminates is killed at the wall-clock timeout.
    assert R.run_pass_at_1("import time\ntime.sleep(30)",
                           ["assert True"], timeout_s=1) is False


def test_run_pass_at_1_assert_test_pass_and_fail():
    assert R.run_pass_at_1("x = 5", ["assert x == 5"]) is True
    assert R.run_pass_at_1("x = 5", ["assert x == 6"]) is False


def test_run_pass_at_1_unknown_dict_shape_uses_test_field():
    assert R.run_pass_at_1("y = 3", [{"test": "assert y == 3"}]) is True


def test_run_pass_at_1_tuple_stdin_stdout():
    assert R.run_pass_at_1("print(int(input()) * 2)", [("4", "8")]) is True


def test_run_pass_at_1_dict_stdin_stdout_and_mismatch():
    ok = R.run_pass_at_1("print(int(input()) + 1)",
                         [{"stdin": "10", "expected_stdout": "11"}])
    assert ok is True
    bad = R.run_pass_at_1("print('nope')",
                          [{"input": "10", "output": "11"}])
    assert bad is False


def test_run_pass_at_1_non_test_object_is_false():
    assert R.run_pass_at_1("x = 1", [123]) is False


def test_run_pass_at_1_functional_call():
    ok = R.run_pass_at_1(
        "def add_one(a):\n    return a + 1\n",
        [{"input": "1", "output": "2", "testtype": "functional"}],
        fn_name="add_one",
    )
    assert ok is True


def test_run_pass_at_1_functional_solution_method():
    ok = R.run_pass_at_1(
        "class Solution:\n    def twice(self, a):\n        return a * 2\n",
        [{"input": "3", "output": "6", "testtype": "functional"}],
        fn_name="twice",
    )
    assert ok is True


# --------------------------------------------------------------------------- #
# functional-value parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_functional_value_json_literal_and_raw():
    assert R._parse_functional_value("[1, 2]") == [1, 2]     # JSON
    assert R._parse_functional_value("(1, 2)") == (1, 2)     # python literal
    assert R._parse_functional_value("") is None             # empty -> None
    assert R._parse_functional_value("not@json") == "not@json"  # raw fallback


def test_parse_functional_args_splits_lines_and_skips_blanks():
    assert R._parse_functional_args("1\n\n[2, 3]\n") == [1, [2, 3]]


def test_stdout_matches_ignores_trailing_whitespace():
    assert R._stdout_matches("5 \n", "5") is True
    assert R._stdout_matches("5\n6", "5\n7") is False


# --------------------------------------------------------------------------- #
# _terminating_role
# --------------------------------------------------------------------------- #
def _traj(*roles):
    task = Task(task_id="t", benchmark="mmlu", prompt="q", answer="A")
    turns = [
        TurnRecord(turn=i + 1, agent_name="m", role=r, raw_output="x", processed_output="x")
        for i, r in enumerate(roles)
    ]
    return Trajectory(task=task, turns=turns, final_answer="x")


def test_terminating_role_none_when_no_turns():
    assert R._terminating_role(_traj()) is None


def test_terminating_role_is_last_turn_role():
    assert R._terminating_role(_traj(Role.WORKER, Role.VERIFIER)) is Role.VERIFIER
