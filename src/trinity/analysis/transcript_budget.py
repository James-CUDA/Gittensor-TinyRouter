"""Offline transcript-budget diagnostic: what does SPEC §4.5 truncation cost?

``roles.postprocess`` caps each turn's output ``O_k`` at a character budget and,
when it fires, keeps a head and a tail joined by :data:`~trinity.roles.postprocess.ELISION_MARKER`.
Its docstring states the intent of the split plainly — *"Bias slightly toward the
tail so the final answer / verdict is preserved."*

Two things follow from that, and neither is currently observable:

1. **The preservation claim is unverified.** The tail bias is a heuristic. A
   verifier that discusses ``ACCEPT`` early and commits ``VERDICT: REVISE`` in the
   middle of a long output can have its committed verdict elided, and
   ``roles.verifier.parse_verdict`` fail-safes to ``REVISE`` — silently turning a
   correct ACCEPT into a wasted extra turn. Nothing counts how often that happens.
2. **SPEC §4.5's revisit trigger has no instrument.** The SPEC closes the
   post-processing decision with *"Revisit only if transcripts overflow the SLM
   context."* Overflow is the condition that would reopen the design, and nothing
   in the repo measures it.

This module answers both from a run's **turn records** — the ``(role, raw_output,
processed_output)`` triples the session already produces. It reports, per role and
for the pooled union:

* how often truncation **fired** and what fraction of characters it **elided**,
* every verifier turn whose committed verdict **did not survive** truncation,
  decided with the canonical :func:`~trinity.roles.verifier.parse_verdict` rather
  than a re-implemented regex,
* the estimated transcript size against the SLM context window, and
* a ``revisit_recommended`` flag implementing SPEC §4.5's own trigger.

A verdict loss is the expensive failure: it does not corrupt an answer, it burns a
turn and hurts the efficiency term, so it is invisible in accuracy alone.

Pure stdlib/math over plain turn mappings -- no torch, no network, no GPU.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..roles.postprocess import ELISION_MARKER
from ..roles.verifier import parse_verdict

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_CONTEXT_TOKENS",
    "DEFAULT_MAX_CHARS",
    "RoleBudget",
    "TranscriptBudget",
    "VerdictLoss",
    "analyze",
    "analyze_benchmarks",
    "counts_by_role",
    "render",
]

#: The three coordinator roles in canonical order (mirrors ``types.ROLE_ORDER``
#: without importing the torch-adjacent coordinator package).
ROLE_NAMES: tuple[str, ...] = ("thinker", "worker", "verifier")

#: ``roles.postprocess.postprocess``'s own default character budget. Kept here so a
#: report can state the budget a run was measured against.
DEFAULT_MAX_CHARS: int = 8000

#: Qwen3-0.6B's context window in tokens (SPEC §3.2 encoder).
DEFAULT_CONTEXT_TOKENS: int = 32768

#: [OUR CHOICE] chars-per-token used to estimate transcript size without a
#: tokenizer. ~4 chars/token is the usual English-text rule of thumb; it keeps this
#: module dependency-free. Overflow is reported as an *estimate* for that reason,
#: and ``chars_per_token`` is a parameter so a caller with a real tokenizer count
#: can pass the measured ratio instead.
DEFAULT_CHARS_PER_TOKEN: float = 4.0


@dataclass(frozen=True)
class VerdictLoss:
    """A verifier turn whose committed verdict did not survive truncation."""

    index: int
    raw_verdict: str | None
    kept_verdict: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "raw_verdict": self.raw_verdict,
            "kept_verdict": self.kept_verdict,
        }


@dataclass(frozen=True)
class RoleBudget:
    """Truncation statistics for one role (or the pooled union)."""

    role: str
    turns: int = 0
    truncated: int = 0
    raw_chars: int = 0
    kept_chars: int = 0
    max_raw_chars: int = 0

    @property
    def elided_chars(self) -> int:
        """Characters dropped by truncation. Never negative."""
        return max(0, self.raw_chars - self.kept_chars)

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.turns if self.turns else 0.0

    @property
    def elision_rate(self) -> float:
        return self.elided_chars / self.raw_chars if self.raw_chars else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "turns": self.turns,
            "truncated": self.truncated,
            "truncation_rate": self.truncation_rate,
            "raw_chars": self.raw_chars,
            "kept_chars": self.kept_chars,
            "elided_chars": self.elided_chars,
            "elision_rate": self.elision_rate,
            "max_raw_chars": self.max_raw_chars,
        }


@dataclass(frozen=True)
class TranscriptBudget:
    """The full diagnostic for one run."""

    pooled: RoleBudget
    per_role: dict[str, RoleBudget] = field(default_factory=dict)
    verdict_losses: tuple[VerdictLoss, ...] = ()
    est_transcript_tokens: int = 0
    context_tokens: int = DEFAULT_CONTEXT_TOKENS
    max_chars: int = DEFAULT_MAX_CHARS

    @property
    def overflows_context(self) -> bool:
        """Whether the kept transcript is estimated to exceed the SLM context."""
        return self.est_transcript_tokens > self.context_tokens

    @property
    def context_headroom_tokens(self) -> int:
        """Estimated tokens still available. Negative when overflowing."""
        return self.context_tokens - self.est_transcript_tokens

    @property
    def revisit_recommended(self) -> bool:
        """SPEC §4.5's own trigger: *revisit only if transcripts overflow*."""
        return self.overflows_context

    def to_dict(self) -> dict[str, Any]:
        return {
            "pooled": self.pooled.to_dict(),
            "per_role": {r: b.to_dict() for r, b in self.per_role.items()},
            "verdict_losses": [v.to_dict() for v in self.verdict_losses],
            "n_verdict_losses": len(self.verdict_losses),
            "est_transcript_tokens": self.est_transcript_tokens,
            "context_tokens": self.context_tokens,
            "context_headroom_tokens": self.context_headroom_tokens,
            "overflows_context": self.overflows_context,
            "revisit_recommended": self.revisit_recommended,
            "max_chars": self.max_chars,
        }


def _role_of(turn: Mapping[str, Any]) -> str:
    """Normalize a turn's role to a lowercase name.

    Accepts a plain string or anything with a ``.value`` (a ``Role`` enum), so a
    caller can pass ``TurnRecord``-shaped mappings without converting first.
    """
    raw = turn.get("role", "")
    value = getattr(raw, "value", raw)
    return str(value).strip().lower()


def _text(turn: Mapping[str, Any], *keys: str) -> str:
    for k in keys:
        v = turn.get(k)
        if isinstance(v, str):
            return v
    return ""


def _was_truncated(raw: str, kept: str) -> bool:
    """Whether ``postprocess`` truncated this turn.

    The elision marker is the positive signal. A length shortfall alone is not
    sufficient — ``postprocess`` also strips surrounding whitespace — and the
    marker can be absent on a hard head-truncation (when the budget is smaller
    than the marker itself), so both are checked.
    """
    if ELISION_MARKER in kept:
        return True
    return len(kept) < len(raw.strip())


def analyze(
    turns: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> TranscriptBudget:
    """Measure truncation cost across a run's turn records.

    Args:
        turns: turn mappings with ``role``, ``raw_output`` and
            ``processed_output`` (the ``TurnRecord`` field names). ``raw`` /
            ``processed`` are accepted as aliases.
        max_chars: the character budget the run was post-processed with. Recorded
            in the report; it does not affect the measurement, which is taken from
            the outputs themselves.
        context_tokens: SLM context window used for the overflow verdict.
        chars_per_token: divisor for the token estimate. Must be positive.

    Returns:
        A :class:`TranscriptBudget`.

    Raises:
        ValueError: if ``chars_per_token`` is not positive or ``context_tokens``
            is negative.
    """
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be > 0, got {chars_per_token!r}")
    if context_tokens < 0:
        raise ValueError(f"context_tokens must be >= 0, got {context_tokens!r}")

    stats: dict[str, dict[str, int]] = {}
    losses: list[VerdictLoss] = []
    total_kept = 0

    for i, turn in enumerate(turns):
        role = _role_of(turn)
        raw = _text(turn, "raw_output", "raw")
        kept = _text(turn, "processed_output", "processed")
        raw_stripped = raw.strip()

        acc = stats.setdefault(
            role,
            {"turns": 0, "truncated": 0, "raw_chars": 0, "kept_chars": 0, "max_raw": 0},
        )
        acc["turns"] += 1
        acc["raw_chars"] += len(raw_stripped)
        acc["kept_chars"] += len(kept)
        acc["max_raw"] = max(acc["max_raw"], len(raw_stripped))
        total_kept += len(kept)

        truncated = _was_truncated(raw, kept)
        if truncated:
            acc["truncated"] += 1

        # Only a truncated verifier turn can lose a verdict; comparing on
        # untruncated turns would flag ordinary parse fail-safes as losses.
        if truncated and role == "verifier":
            raw_v = parse_verdict(raw_stripped)
            kept_v = parse_verdict(kept)
            if raw_v != kept_v:
                losses.append(VerdictLoss(index=i, raw_verdict=raw_v, kept_verdict=kept_v))

    per_role = {
        r: RoleBudget(
            role=r,
            turns=a["turns"],
            truncated=a["truncated"],
            raw_chars=a["raw_chars"],
            kept_chars=a["kept_chars"],
            max_raw_chars=a["max_raw"],
        )
        for r, a in sorted(stats.items())
    }
    pooled = RoleBudget(
        role="all",
        turns=sum(b.turns for b in per_role.values()),
        truncated=sum(b.truncated for b in per_role.values()),
        raw_chars=sum(b.raw_chars for b in per_role.values()),
        kept_chars=sum(b.kept_chars for b in per_role.values()),
        max_raw_chars=max((b.max_raw_chars for b in per_role.values()), default=0),
    )
    return TranscriptBudget(
        pooled=pooled,
        per_role=per_role,
        verdict_losses=tuple(losses),
        est_transcript_tokens=int(total_kept / chars_per_token),
        context_tokens=context_tokens,
        max_chars=max_chars,
    )


def analyze_benchmarks(
    by_benchmark: Mapping[str, Iterable[Mapping[str, Any]]],
    **kwargs: Any,
) -> dict[str, TranscriptBudget]:
    """Run :func:`analyze` per benchmark, plus a pooled ``"all"`` entry.

    The pooled entry re-analyzes the concatenated turns rather than averaging the
    per-benchmark reports, so rates stay weighted by turn count.
    """
    materialized = {k: list(v) for k, v in by_benchmark.items()}
    out = {k: analyze(v, **kwargs) for k, v in sorted(materialized.items())}
    if materialized:
        pooled_turns: list[Mapping[str, Any]] = []
        for _k, v in sorted(materialized.items()):
            pooled_turns.extend(v)
        out["all"] = analyze(pooled_turns, **kwargs)
    return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render(
    turns: Iterable[Mapping[str, Any]] | None = None,
    *,
    report: TranscriptBudget | None = None,
    **kwargs: Any,
) -> str:
    """Render a human-readable report.

    Pass either ``turns`` (analyzed here) or a precomputed ``report``.

    Raises:
        ValueError: if neither or both of ``turns`` / ``report`` are given.
    """
    if (turns is None) == (report is None):
        raise ValueError("pass exactly one of turns= or report=")
    rep = report if report is not None else analyze(turns or [], **kwargs)

    lines = [
        f"transcript budget (max_chars={rep.max_chars})",
        f"  turns          : {rep.pooled.turns}",
        f"  truncated      : {rep.pooled.truncated} ({_pct(rep.pooled.truncation_rate)})",
        f"  chars elided   : {rep.pooled.elided_chars} ({_pct(rep.pooled.elision_rate)})",
        f"  longest output : {rep.pooled.max_raw_chars} chars",
    ]
    for role in ROLE_NAMES:
        b = rep.per_role.get(role)
        if b is None:
            continue
        lines.append(
            f"    {role:9s}: {b.truncated}/{b.turns} truncated "
            f"({_pct(b.truncation_rate)}), {_pct(b.elision_rate)} of chars elided"
        )
    for role in sorted(set(rep.per_role) - set(ROLE_NAMES)):
        b = rep.per_role[role]
        lines.append(
            f"    {role:9s}: {b.truncated}/{b.turns} truncated "
            f"({_pct(b.truncation_rate)}), {_pct(b.elision_rate)} of chars elided"
        )

    lines.append(
        f"  est. transcript: ~{rep.est_transcript_tokens} tokens vs "
        f"{rep.context_tokens} context (headroom {rep.context_headroom_tokens})"
    )
    if rep.verdict_losses:
        lines.append(f"  VERDICT LOSSES : {len(rep.verdict_losses)}")
        for v in rep.verdict_losses:
            lines.append(
                f"    turn {v.index}: {v.raw_verdict!r} -> {v.kept_verdict!r} after truncation"
            )
    else:
        lines.append("  verdict losses : none — the tail bias held on every truncated verifier turn")

    if rep.revisit_recommended:
        lines.append(
            "  SPEC §4.5: transcripts overflow the SLM context — REVISIT post-processing"
        )
    else:
        lines.append("  SPEC §4.5: no overflow — the pass-through policy still holds")
    return "\n".join(lines)


def counts_by_role(turns: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Turn counts per role. Small helper for callers building their own views."""
    return dict(Counter(_role_of(t) for t in turns))
