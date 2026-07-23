"""BBH exact-match must accept a Markdown-emphasised correct answer.

Bolding the final answer (``Answer: **True**``) is a very common model format —
common enough that the shared choice extractor grew ``_strip_choice_md_emphasis``
for it. But BBH's ``_normalize_exact`` strip set had no ``*``/``_``, so
``**True**`` normalised to ``**true**`` and every exact-match subtask
(boolean_expressions, formal_fallacies, navigate, sports_understanding,
object_counting, ...) graded the bolded correct answer 0.0 — while the
multiple-choice half of the SAME adapter accepted ``Answer: **B**``.

The fix peels emphasis only when the SAME marker wraps the whole answer, so
asymmetric or inner-word markers never alter content, and dyck bracket
sequences still compare as sequences. No network, no GPU, pure text.
"""
from __future__ import annotations

from trinity.adapters.bbh import _normalize_exact, score_bbh

_EXACT = {"answer": "True", "answer_type": "exact_match", "subtask": "boolean_expressions"}


# --- normalization ---


def test_bold_and_italic_unwrap_to_content():
    assert _normalize_exact("**True**") == "true"
    assert _normalize_exact("*valid*") == "valid"
    assert _normalize_exact("__no__") == "no"
    assert _normalize_exact("**_True_**") == "true"


def test_emphasis_inside_terminal_punctuation():
    assert _normalize_exact("**True**.") == "true"
    assert _normalize_exact('"**valid**"') == "valid"


def test_multiword_answer_unwraps():
    assert _normalize_exact("**not plausible**") == "not plausible"


def test_asymmetric_marker_is_not_peeled():
    # An asymmetric "*" is content, not a wrapper — unchanged behavior.
    assert _normalize_exact("2*3") == "2*3"


def test_dyck_bracket_sequence_still_compares_as_sequence():
    assert _normalize_exact("] )") == "])"
    # ...even when a model bolds it.
    assert _normalize_exact("**] )**") == "])"


# --- grading: end-to-end through score_bbh ---


def test_bolded_correct_answer_scores_one():
    # Was 0.0 before the fix while the plain form scored 1.0.
    assert score_bbh("Answer: True", _EXACT) == 1.0
    assert score_bbh("Answer: **True**", _EXACT) == 1.0


def test_bolded_wrong_answer_still_scores_zero():
    assert score_bbh("Answer: **False**", _EXACT) == 0.0


def test_valid_invalid_subtask():
    ref = {"answer": "valid", "answer_type": "exact_match", "subtask": "formal_fallacies"}
    assert score_bbh("Answer: **valid**", ref) == 1.0
    assert score_bbh("Answer: **invalid**", ref) == 0.0
