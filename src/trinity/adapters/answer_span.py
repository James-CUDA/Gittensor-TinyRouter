"""Shared "pull the final answer out of a chatty output" extractor.

Both the DROP and BIG-Bench-Hard adapters prompt for chain-of-thought and ask the
model to *end* with an explicit lead — ``"Reason over the passage, then end with
'Answer: <answer>'"`` / ``"Think step by step, then end with 'Answer: <answer>'"``.
The reasoning that precedes it routinely contains its own ``"answer is"`` /
``"answer:"`` phrasing ("To answer: we count the houses...", "The answer is the
total of the two years."), so the extractor must take the **last** lead, not the
first — final answers come last.

This module is the single source of truth for that rule; ``drop.py`` and ``bbh.py``
both use it instead of carrying private (previously byte-identical) copies.

Pure / deterministic / no network / no GPU.
"""
from __future__ import annotations

import re

__all__ = ["ANSWER_LEAD", "final_answer_segment"]

#: An explicit answer lead: ``answer is <x>`` / ``answer: <x>`` / ``final answer: <x>``.
#: Only the remainder of the *same line* is kept, so this is deliberately not
#: ``DOTALL`` — the answer proper never spans lines.
ANSWER_LEAD = re.compile(r"(?:final\s+)?answer\s*(?:is|:)\s*(.*)", re.IGNORECASE)


def final_answer_segment(text: str) -> str:
    """Return the answer portion of a (possibly chatty) model output.

    Prefers the text after the **last** explicit ``"answer is"`` / ``"answer:"``
    lead that is actually followed by something — final answers come last, and the
    chain-of-thought both adapters request frequently contains an earlier
    ``"answer ..."`` phrase that must not hijack the extraction. A lead with
    nothing after it (a dangling ``"Answer:"``) is skipped in favour of an earlier
    real one. With no usable lead, falls back to the last non-empty line.

    Args:
        text: The raw model output.

    Returns:
        The extracted answer segment (first line of the captured span, stripped),
        or ``""`` when ``text`` is empty or yields nothing.
    """
    if not text:
        return ""
    for match in reversed(list(ANSWER_LEAD.finditer(text))):
        seg = match.group(1).strip()
        if seg:
            return seg
    last_line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
    return last_line.strip()
