"""Regression: a null-valued text PART must normalize to "", not "None".

The earlier fix handled a top-level ``content: null``; this covers the list
branch, where a part ``{"type": "text", "text": null}`` went through
``str(item.get("text", ""))`` -> ``str(None)`` -> "None" and spliced that literal
into ``ChatResult.text`` (and thus the transcript). No network, no GPU.
"""
from __future__ import annotations

from trinity.llm.openrouter_client import _message_text


def test_null_valued_text_part_is_empty():
    assert _message_text({"content": [{"type": "text", "text": None}]}) == ""


def test_null_text_part_does_not_corrupt_real_text():
    msg = {"content": [
        {"type": "text", "text": "The answer is 4."},
        {"type": "text", "text": None},
    ]}
    assert _message_text(msg) == "The answer is 4."


def test_missing_text_key_still_empty():
    # The old default path (key absent) must stay correct.
    assert _message_text({"content": [{"type": "text"}]}) == ""


def test_normal_text_parts_unchanged():
    msg = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _message_text(msg) == "ab"


def test_non_text_and_null_parts_are_skipped_together():
    msg = {"content": [
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": None},
        {"type": "text", "text": "kept"},
    ]}
    assert _message_text(msg) == "kept"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
