"""The shared final-answer extractor takes the LAST answer lead.

Both the DROP and BBH prompts ask the model to reason first and only then end with
``"Answer: <answer>"``. That chain-of-thought routinely contains its own
``"answer is"`` / ``"answer:"`` phrasing, which must not hijack the extraction —
final answers come last.

Pure / offline: no torch, no network.
"""
from __future__ import annotations

import pytest

from trinity.adapters.answer_span import final_answer_segment
from trinity.adapters.bbh import score_bbh
from trinity.adapters.drop import score_drop


# --------------------------------------------------------------------------- #
# The extractor itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        # An earlier "answer ..." in the reasoning must not win.
        ("To answer: we count the houses.\nAnswer: 21", "21"),
        ("The answer is the total of the two years.\n12 + 9 = 21.\nAnswer: 21", "21"),
        ("Let us reason. The answer: depends.\nAnswer: True", "True"),
        # A single lead still works.
        ("12 + 9 = 21.\nAnswer: 21", "21"),
        ("so the final answer is True.", "True."),
        # No lead -> last non-empty line.
        ("reasoning line\nTrue", "True"),
        ("last line wins\n\n", "last line wins"),
        # Empty input.
        ("", ""),
    ],
)
def test_final_answer_segment_takes_the_last_lead(text, expected):
    assert final_answer_segment(text) == expected


def test_dangling_lead_falls_back_to_an_earlier_real_lead():
    # A trailing bare "Answer:" carries nothing; the real answer precedes it.
    assert final_answer_segment("Answer: 21\nAnswer:") == "21"


def test_only_the_answer_line_is_kept():
    assert final_answer_segment("Answer: 21\ntrailing commentary") == "21"


# --------------------------------------------------------------------------- #
# End-to-end: chain-of-thought answers score correctly in both adapters
# --------------------------------------------------------------------------- #
def test_drop_chain_of_thought_answer_scores_correct():
    cot = "To answer: we count the houses. 12 were built in 2018 and 9 in 2019.\nAnswer: 21"
    assert score_drop(cot, {"gold_answers": ["21"]}) == 1.0


def test_drop_answer_is_phrase_in_reasoning_does_not_hijack():
    cot = "The answer is the total of the two years.\n12 + 9 = 21.\nAnswer: 21"
    assert score_drop(cot, {"gold_answers": ["21"]}) == 1.0


def test_bbh_chain_of_thought_answer_scores_correct():
    cot = ("Let us reason. The answer: depends on precedence.\n"
           "not (True and False) = not False = True.\nAnswer: True")
    assert score_bbh(cot, {"answer": "True", "answer_type": "exact_match"}) == 1.0


def test_bbh_wrong_answer_is_still_zero():
    # The fix must not turn a wrong answer into a pass.
    cot = "The answer is tricky.\nAnswer: False"
    assert score_bbh(cot, {"answer": "True", "answer_type": "exact_match"}) == 0.0
