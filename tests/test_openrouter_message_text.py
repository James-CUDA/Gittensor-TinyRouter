"""Normalization of OpenRouter/OpenAI message content to plain text.

Regression cover for the bug where ``{"content": None}`` normalized to the literal
4-character string ``"None"`` rather than ``""``. ``dict.get(key, default)`` only
substitutes the default for an *absent* key; a present-but-null ``content`` fell
through both ``isinstance`` branches to ``str(content)``.

That value becomes ``ChatResult.text``, which feeds the multi-turn transcript the
coordinator conditions on (``orchestration/session.py``) and the reward scorer
(``eval.py``) -- so a null completion was graded as if the model had answered with
the word "None".

Offline: no network, no client construction.
"""
from __future__ import annotations

import pytest

from trinity.llm.openrouter_client import _message_text


# --------------------------------------------------------------------------- #
# The bug: null content is the empty string, not "None"
# --------------------------------------------------------------------------- #
def test_null_content_is_empty_string():
    """The regression: previously returned the literal 'None'."""
    assert _message_text({"content": None}) == ""


def test_null_content_is_never_the_literal_none_string():
    """Guard the specific corruption, not just the emptiness."""
    assert _message_text({"content": None}) != "None"


def test_null_content_with_reasoning_only_turn():
    """Reasoning models routinely emit a turn with reasoning but no content."""
    msg = {"role": "assistant", "content": None, "reasoning": "thinking..."}
    assert _message_text(msg) == ""


def test_null_content_with_tool_calls_only_turn():
    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}
    assert _message_text(msg) == ""


def test_null_and_empty_content_are_indistinguishable_downstream():
    """A truncated completion must not fabricate content an empty one lacks."""
    assert _message_text({"content": None}) == _message_text({"content": ""})


# --------------------------------------------------------------------------- #
# Existing behaviour must not regress
# --------------------------------------------------------------------------- #
def test_absent_content_key_is_empty_string():
    assert _message_text({"role": "assistant"}) == ""


def test_empty_string_content_round_trips():
    assert _message_text({"content": ""}) == ""


def test_plain_string_content_round_trips():
    assert _message_text({"content": "hello"}) == "hello"


def test_whitespace_content_is_preserved_verbatim():
    """Trimming is the caller's job; the normalizer must not silently strip."""
    assert _message_text({"content": "  spaced  "}) == "  spaced  "


# --------------------------------------------------------------------------- #
# Structured (list) content
# --------------------------------------------------------------------------- #
def test_list_content_concatenates_text_parts():
    msg = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _message_text(msg) == "ab"


def test_list_content_skips_non_text_parts():
    msg = {"content": [{"type": "image_url"}, {"type": "text", "text": "only"}]}
    assert _message_text(msg) == "only"


def test_list_content_skips_non_dict_items():
    msg = {"content": [None, "raw", {"type": "text", "text": "kept"}]}
    assert _message_text(msg) == "kept"


def test_empty_list_content_is_empty_string():
    assert _message_text({"content": []}) == ""


def test_list_of_only_non_text_parts_is_empty_string():
    assert _message_text({"content": [{"type": "image_url"}]}) == ""


# --------------------------------------------------------------------------- #
# Unexpected types still degrade to str(), as before
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weird", [123, 4.5, True, {"unexpected": "dict"}])
def test_unexpected_type_falls_back_to_str(weird):
    assert _message_text({"content": weird}) == str(weird)
