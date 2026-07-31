"""Audit an on-disk LLM response cache: size, per-model dollar value, and pollution.

``trinity.llm.cache`` is a content-addressed completion cache, but its :class:`CacheStats`
is **in-memory and per-run** — reset every process, never persisted. Nothing reads an
*existing* cache directory, so a contributor who runs many eval sweeps with
``TRINITY_LLM_CACHE=...`` has no visibility into what the cache is worth or whether it is
polluted.

This walks a committed cache directory (the sharded ``root/<2 hex>/<key>.json`` layout
``ResponseCache.path_for`` writes) and reports, per model, the entry count and prompt /
completion tokens, plus the **dollar value on re-serve** — the money the cache saves if
every entry is hit once, priced through :func:`trinity.llm.openrouter_pricing.token_cost`.

It also counts **pollution**: entries that violate the cache's own
``_is_cacheable_result`` invariant (``finish_reason == "error"`` or blank ``text``). Such
a record — if written by an older/buggy writer — silently re-serves a permanent 0 score
for that item, so surfacing them is correctness-adjacent. Records with a stale schema
version (which ``ResponseCache.get`` already ignores) and unreadable files are counted
separately. Read-only, pure stdlib + the pricing table — no torch, no network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trinity.llm.cache import _SCHEMA_VERSION
from trinity.llm.openrouter_pricing import token_cost

__all__ = [
    "ModelCacheStats",
    "CacheAudit",
    "audit_cache",
    "render",
]

#: Cap the sample of polluted keys kept in the report (the count is always exact).
_MAX_POLLUTED_SAMPLE = 20


def _int(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class ModelCacheStats:
    """One model's contribution to the cache."""

    model: str
    entries: int
    prompt_tokens: int
    completion_tokens: int
    dollar_value: float          # token_cost over this model's cached tokens

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "entries": self.entries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "dollar_value": self.dollar_value,
        }


@dataclass(frozen=True)
class CacheAudit:
    """A read-only audit of an on-disk response cache directory."""

    root: str
    n_files: int
    n_valid: int                 # current-schema records (what get() would serve)
    n_stale: int                 # wrong schema version (get() ignores these)
    n_unreadable: int            # not JSON / not a JSON object
    n_polluted: int              # valid records violating _is_cacheable_result
    per_model: list[ModelCacheStats]     # sorted by dollar_value desc
    total_prompt_tokens: int
    total_completion_tokens: int
    total_dollar_value: float
    polluted_keys: list[str]     # up to _MAX_POLLUTED_SAMPLE example keys

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "root": self.root,
            "n_files": self.n_files,
            "n_valid": self.n_valid,
            "n_stale": self.n_stale,
            "n_unreadable": self.n_unreadable,
            "n_polluted": self.n_polluted,
            "per_model": [m.to_dict() for m in self.per_model],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_dollar_value": self.total_dollar_value,
            "polluted_keys": list(self.polluted_keys),
        }


def _is_polluted(record: dict[str, Any]) -> bool:
    """True iff a valid record should never have been cached (error / blank text).

    Mirrors ``cache._is_cacheable_result`` inverted: an ``error`` finish reason or an
    empty/whitespace ``text`` is a transient failure that must not be re-served.
    """
    if record.get("finish_reason") == "error":
        return True
    return not str(record.get("text", "") or "").strip()


def audit_cache(root: str | Path) -> CacheAudit:
    """Walk a response-cache directory and summarize its contents.

    Args:
        root: The cache directory (``$TRINITY_LLM_CACHE``). A missing directory yields
            an empty audit rather than raising.

    Returns:
        A :class:`CacheAudit`. Validity matches ``ResponseCache.get``: a record counts as
        ``valid`` only when it is a JSON object with the current ``_SCHEMA_VERSION``.
    """
    root = Path(root)
    files = sorted(root.rglob("*.json")) if root.is_dir() else []

    per_model: dict[str, dict[str, float]] = {}
    n_valid = n_stale = n_unreadable = n_polluted = 0
    polluted_keys: list[str] = []
    for path in files:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            n_unreadable += 1
            continue
        if not isinstance(record, dict):
            n_unreadable += 1
            continue
        if record.get("v") != _SCHEMA_VERSION:
            n_stale += 1
            continue
        n_valid += 1
        model = str(record.get("model", "") or "?")
        pt, ct = _int(record.get("prompt_tokens")), _int(record.get("completion_tokens"))
        agg = per_model.setdefault(
            model, {"entries": 0, "prompt_tokens": 0, "completion_tokens": 0, "dollar": 0.0})
        agg["entries"] += 1
        agg["prompt_tokens"] += pt
        agg["completion_tokens"] += ct
        agg["dollar"] += token_cost(model, pt, ct)
        if _is_polluted(record):
            n_polluted += 1
            if len(polluted_keys) < _MAX_POLLUTED_SAMPLE:
                polluted_keys.append(path.stem)

    models = [
        ModelCacheStats(
            model=m, entries=int(a["entries"]),
            prompt_tokens=int(a["prompt_tokens"]), completion_tokens=int(a["completion_tokens"]),
            dollar_value=a["dollar"],
        )
        for m, a in per_model.items()
    ]
    models.sort(key=lambda m: m.dollar_value, reverse=True)
    return CacheAudit(
        root=str(root),
        n_files=len(files),
        n_valid=n_valid,
        n_stale=n_stale,
        n_unreadable=n_unreadable,
        n_polluted=n_polluted,
        per_model=models,
        total_prompt_tokens=sum(m.prompt_tokens for m in models),
        total_completion_tokens=sum(m.completion_tokens for m in models),
        total_dollar_value=sum(m.dollar_value for m in models),
        polluted_keys=polluted_keys,
    )


def render(audit: CacheAudit) -> str:
    """Markdown: per-model value table + the totals and pollution/staleness summary."""
    a = audit
    out = [f"# LLM cache audit — {a.root}\n"]
    if a.n_files == 0:
        return "".join(out) + "\n_(cache directory is empty or absent)_\n"

    out.append("| model | entries | prompt tok | completion tok | value on re-serve |")
    out.append("|---|---|---|---|---|")
    for m in a.per_model:
        out.append(f"| {m.model} | {m.entries} | {m.prompt_tokens:,} | "
                   f"{m.completion_tokens:,} | ${m.dollar_value:.4f} |")
    out.append(f"| **total** | **{a.n_valid}** | **{a.total_prompt_tokens:,}** | "
               f"**{a.total_completion_tokens:,}** | **${a.total_dollar_value:.4f}** |")

    out.append(f"\n- **{a.n_valid}** valid entries worth **${a.total_dollar_value:.4f}** "
               f"on re-serve, across {a.n_files} files")
    if a.n_stale:
        out.append(f"- {a.n_stale} stale-schema entries (ignored on read)")
    if a.n_unreadable:
        out.append(f"- {a.n_unreadable} unreadable files")
    if a.n_polluted:
        sample = ", ".join(k[:12] for k in a.polluted_keys)
        out.append(f"- ⚠ **{a.n_polluted} polluted entries** (error/blank — would re-serve "
                   f"a 0 score): {sample}{' …' if a.n_polluted > len(a.polluted_keys) else ''}")
    else:
        out.append("- no polluted entries ✅")
    return "\n".join(out) + "\n"
