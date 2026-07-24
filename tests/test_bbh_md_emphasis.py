"""BBH exact-match must accept Markdown-emphasised answers (#461)."""

from __future__ import annotations

from trinity.adapters.bbh import score_bbh


def test_exact_match_accepts_markdown_bold() -> None:
    ref = {"answer": "True", "answer_type": "exact_match", "subtask": "boolean_expressions"}
    assert score_bbh("Answer: True", ref) == 1.0
    assert score_bbh("Answer: **True**", ref) == 1.0
    assert score_bbh("Answer: **valid**", {"answer": "valid", "answer_type": "exact_match"}) == 1.0
