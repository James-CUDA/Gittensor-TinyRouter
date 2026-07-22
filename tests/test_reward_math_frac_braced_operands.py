"""Fraction unwrapping must scan balanced braces, not a single brace level.

``normalize_math_answer`` rewrote ``\\frac{a}{b}`` with a regex whose operands
were ``[^{}]+``, so any fraction carrying a braced sub-expression — a radical, a
nested fraction, a braced exponent — kept its command name verbatim and matched
nothing (issue #409). ``\\frac{\\sqrt{2}}{2}`` is one of the most common MATH-500
answer shapes, so this was a systematic false negative on a competition
benchmark.

These tests pin the fixed behaviour:

* braced operands unwrap, so ``\\frac``/``\\dfrac``/``\\tfrac`` stay interchangeable
  at any nesting depth (the #145 invariant, which the old regex silently dropped);
* the result is parenthesis-free where precedence allows, so it matches a
  reference written with a plain slash;
* precedence is still respected, so no false POSITIVE is introduced;
* nothing here depends on the optional sympy fallback.

Pure / offline — no torch, no network.
"""
from __future__ import annotations

import re

from trinity.orchestration.reward import math_equal, normalize_math_answer, score_text


# ---------------------------------------------------------------------------
# A braced operand unwraps instead of surviving as a literal command
# ---------------------------------------------------------------------------
def test_radical_numerator_unwraps_to_plain_slash():
    # Must be exactly "sqrt(2)/2" — a stray "(sqrt(2))/(2)" would still miss a
    # reference written \sqrt{2}/2, which is the whole point of the fix.
    assert normalize_math_answer(r"\frac{\sqrt{2}}{2}") == "sqrt(2)/2"
    assert normalize_math_answer(r"\frac{2\sqrt{3}}{3}") == "2sqrt(3)/3"
    assert normalize_math_answer(r"\frac{1}{\sqrt{2}}") == "1/sqrt(2)"


def test_no_latex_command_survives_normalization():
    for src in (r"\frac{\sqrt{2}}{2}", r"\dfrac{\sqrt{3}}{3}", r"\tfrac{\pi}{4}"):
        assert "frac" not in normalize_math_answer(src)


def test_nested_fraction_in_numerator_folds_to_a_single_ratio():
    # \frac{\frac{1}{2}}{3} renders as 1/6.
    assert normalize_math_answer(r"\frac{\frac{1}{2}}{3}") == "1/6"


def test_nested_fraction_in_denominator_stays_grouped():
    # 1/(2/3) is 3/2, NOT 1/2/3 == 1/6. The grouping must survive normalization;
    # reducing it is left to the numeric/symbolic comparison in math_equal.
    assert normalize_math_answer(r"\frac{1}{\frac{2}{3}}") == "1/(2/3)"
    assert math_equal(r"\frac{1}{\frac{2}{3}}", "1/6") is False


def test_fraction_inside_a_radicand_unwraps():
    # \sqrt's radicand regex is brace-free, so this needs \frac to run first —
    # the opposite order from the case above.
    assert normalize_math_answer(r"\sqrt{\frac{1}{2}}") == "sqrt(1/2)"


# ---------------------------------------------------------------------------
# The \frac / \dfrac / \tfrac family stays interchangeable (issue #145 invariant)
# ---------------------------------------------------------------------------
def test_fraction_family_agrees_on_a_braced_payload():
    forms = [normalize_math_answer(f"\\{cmd}{{\\sqrt{{2}}}}{{2}}") for cmd in ("frac", "dfrac", "tfrac")]
    assert len(set(forms)) == 1, forms
    assert forms[0] == "sqrt(2)/2"


# ---------------------------------------------------------------------------
# End-to-end grading of a correct answer
# ---------------------------------------------------------------------------
def test_radical_over_integer_scores_correct():
    assert score_text("math500", r"So the answer is \boxed{\frac{\sqrt{2}}{2}}.", r"\sqrt{2}/2") == 1.0
    assert score_text("math500", r"\boxed{\tfrac{\sqrt{2}}{2}}", r"\frac{\sqrt{2}}{2}") == 1.0
    assert score_text("math500", r"\boxed{\frac{2\sqrt{3}}{3}}", r"2\sqrt{3}/3") == 1.0
    assert score_text("math500", r"\boxed{\frac{\frac{1}{2}}{3}}", "1/6") == 1.0


def test_wrong_radical_answer_is_still_wrong():
    assert score_text("math500", r"\boxed{\frac{\sqrt{3}}{2}}", r"\sqrt{2}/2") == 0.0
    assert score_text("math500", r"\boxed{\frac{\sqrt{2}}{3}}", r"\sqrt{2}/2") == 0.0


# ---------------------------------------------------------------------------
# Precedence is preserved — dropping parentheses must not create false positives
# ---------------------------------------------------------------------------
def test_compound_denominator_keeps_its_parentheses():
    # 1/(2sqrt(3)) is NOT (1/2)*sqrt(3); the denominator must stay grouped.
    assert normalize_math_answer(r"\frac{1}{2\sqrt{3}}") == "1/(2sqrt(3))"
    assert math_equal(r"\frac{1}{2\sqrt{3}}", r"1/2\sqrt{3}") is False


def test_additive_numerator_keeps_its_parentheses():
    assert normalize_math_answer(r"\frac{a+b}{c}") == "(a+b)/c"
    assert math_equal(r"\frac{a+b}{c}", "a+b/c") is False


# ---------------------------------------------------------------------------
# Previously-working forms are unchanged
# ---------------------------------------------------------------------------
def test_brace_free_fractions_are_unchanged():
    assert normalize_math_answer(r"\frac{1}{2}") == "1/2"
    assert normalize_math_answer(r"\dfrac{3}{4}") == "3/4"
    assert normalize_math_answer(r"\tfrac{3}{4}") == "3/4"
    assert normalize_math_answer(r"-\frac{3}{4}") == "-3/4"
    assert normalize_math_answer(r"\frac12") == "1/2"
    assert normalize_math_answer(r"\frac{\pi}{2}") == "pi/2"


def test_cfrac_is_still_left_untouched():
    # \cfrac (continued fraction) is not a plain a/b and must not be rewritten.
    assert normalize_math_answer(r"\cfrac{3}{4}") == r"\cfrac{3}{4}"


# ---------------------------------------------------------------------------
# Malformed input terminates and is left alone
# ---------------------------------------------------------------------------
def test_unbalanced_or_operandless_fraction_is_left_untouched():
    for src in (r"\frac", r"\frac{1}", r"\frac{1}{2", r"\frac{{1}{2}"):
        assert normalize_math_answer(src) == src


def test_fix_does_not_depend_on_sympy(monkeypatch):
    # The normalizer must resolve these by string rewriting alone; sympy is an
    # optional dependency and _sympy_equal returns False without it.
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sympy" or name.startswith("sympy."):
            raise ImportError("sympy disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert score_text("math500", r"\boxed{\frac{\sqrt{2}}{2}}", r"\sqrt{2}/2") == 1.0
    assert score_text("math500", r"\boxed{\frac{\frac{1}{2}}{3}}", "1/6") == 1.0


# ---------------------------------------------------------------------------
# The regression guard: no single-brace-level fraction regex may come back
# ---------------------------------------------------------------------------
def test_normalizer_does_not_use_a_brace_free_frac_regex():
    from trinity.orchestration import reward

    source = re.sub(r"#.*", "", __import__("inspect").getsource(reward.normalize_math_answer))
    assert r"frac\s*\{([^{}]+)\}" not in source
