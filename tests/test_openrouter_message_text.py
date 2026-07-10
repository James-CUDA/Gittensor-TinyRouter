"""Offline tests for OpenRouter message-content normalization. No network, no GPU.

`message.content` is nullable in the chat-completions schema. `_message_text` must
normalize every "no text" shape to `""`, because its result becomes
`ChatResult.text` and is appended verbatim to the trajectory transcript.
"""
from __future__ import annotations

from trinity.llm.openrouter_client import _message_text


# ---------------------------------------------------------------------------
# Regression: a null completion must not become the literal string "None"
# ---------------------------------------------------------------------------
def test_null_content_normalizes_to_empty_string():
    # Providers send content=null for an empty completion, or when the message
    # carries only reasoning / tool-call metadata. str(None) == "None" would
    # inject a fake model utterance into the transcript.
    assert _message_text({"content": None}) == ""


def test_all_empty_shapes_agree():
    # Missing key, empty string, empty list and null must all mean "no text".
    assert _message_text({}) == ""
    assert _message_text({"content": ""}) == ""
    assert _message_text({"content": []}) == ""
    assert _message_text({"content": None}) == ""


# ---------------------------------------------------------------------------
# Existing behaviour is preserved
# ---------------------------------------------------------------------------
def test_plain_string_content_passes_through():
    assert _message_text({"content": "The answer is 4."}) == "The answer is 4."


def test_structured_content_concatenates_text_parts():
    content = [
        {"type": "text", "text": "The answer "},
        {"type": "text", "text": "is 4."},
    ]
    assert _message_text({"content": content}) == "The answer is 4."


def test_structured_content_skips_non_text_and_non_dict_items():
    content = [
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        "not-a-dict",
        {"type": "text", "text": "kept"},
    ]
    assert _message_text({"content": content}) == "kept"


def test_reasoning_only_message_yields_no_text():
    # A reasoning-only assistant message: content is null, reasoning is elsewhere.
    msg = {"content": None, "reasoning": "thinking out loud", "role": "assistant"}
    assert _message_text(msg) == ""


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
