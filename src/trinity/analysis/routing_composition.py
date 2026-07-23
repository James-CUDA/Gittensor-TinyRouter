"""Offline routing-composition diagnostic: what does the coordinator actually pick?

The submission gate ``audit_head_routing_diversity`` (gate 10) inspects a head's
*weights* and warns when the agent logit rows collapse to one model. That is a
static check on ``W`` alone. It says nothing about what the head does on *real
inputs*: a head whose weights look diverse can still, on a given benchmark, route
almost everything to one model or never call the Verifier.

This module answers that empirical question from a coordinator's **decision log** —
the ``(model, role)`` picks it actually made across a benchmark's turns. It reports,
per benchmark and for the pooled union:

* the usage **share** of each model and of each role (Thinker / Worker / Verifier),
* which pool models / roles were **never used**,
* the **normalized entropy** of the model distribution (1.0 = uniform, 0.0 = one
  model) as a single collapse-o-meter, and
* a **collapsed** flag when one model's share is at or above ``collapse_threshold``.

A collapsed router is a valid but usually strategically poor submission (it ignores
the query), and a head that never routes to the Verifier cannot benefit from the
multi-turn accept/revise loop — both are worth seeing before a training run, offline
and for free.

Pure stdlib/math over plain ``(model, role)`` pairs -- no torch, no network, no GPU.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "RoutingComposition",
    "analyze",
    "analyze_benchmarks",
    "render",
]

#: The three coordinator roles, in the canonical order (mirrors ``types.ROLE_ORDER``
#: without importing torch-adjacent modules).
ROLE_NAMES: tuple[str, ...] = ("thinker", "worker", "verifier")

#: A model's share at or above this fraction flags the router as collapsed.
_DEFAULT_COLLAPSE_THRESHOLD = 0.90


def _shares(counts: Mapping[str, int], total: int) -> dict[str, float]:
    return {k: (c / total if total else 0.0) for k, c in counts.items()}


def _normalized_entropy(counts: Iterable[int]) -> float:
    """Shannon entropy of a count distribution, normalized to ``[0, 1]``.

    ``1.0`` when the picks are spread uniformly over the observed choices, ``0.0``
    when they all fall on one choice (or there are fewer than two choices). Uses the
    number of *observed* choices as the base, so it measures how evenly the router
    spread its picks over the options it actually used.
    """
    cs = [c for c in counts if c > 0]
    total = sum(cs)
    if total == 0 or len(cs) < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in cs)
    return h / math.log(len(cs))


@dataclass(frozen=True)
class RoutingComposition:
    """Empirical routing composition over one benchmark's ``(model, role)`` picks."""

    benchmark: str
    n_decisions: int
    model_shares: dict[str, float]
    role_shares: dict[str, float]
    unused_models: list[str]
    unused_roles: list[str]
    model_entropy: float
    top_model: str | None
    top_model_share: float
    collapsed: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_decisions": self.n_decisions,
            "model_shares": self.model_shares,
            "role_shares": self.role_shares,
            "unused_models": self.unused_models,
            "unused_roles": self.unused_roles,
            "model_entropy": self.model_entropy,
            "top_model": self.top_model,
            "top_model_share": self.top_model_share,
            "collapsed": self.collapsed,
        }


def _normalize_decisions(decisions: Iterable[Any]) -> list[tuple[str, str | None]]:
    """Coerce decisions to ``(model, role)`` pairs.

    Accepts ``(model, role)`` tuples/lists, a bare model string (role ``None``), or a
    mapping with ``model`` and ``role`` keys. Role is lower-cased; a ``Role`` enum's
    ``.value`` is picked up via ``str``. Empty / unrecognisable entries are skipped.
    """
    out: list[tuple[str, str | None]] = []
    for d in decisions:
        model: Any
        role: Any = None
        if isinstance(d, Mapping):
            model, role = d.get("model"), d.get("role")
        elif isinstance(d, (tuple, list)):
            if not d:
                continue
            model = d[0]
            role = d[1] if len(d) > 1 else None
        else:
            model = d
        if model is None or model == "":
            continue
        role_val = getattr(role, "value", role)
        out.append((str(model), str(role_val).lower() if role_val is not None else None))
    return out


def analyze(
    decisions: Iterable[Any],
    *,
    benchmark: str = "?",
    pool_models: Sequence[str] | None = None,
    collapse_threshold: float = _DEFAULT_COLLAPSE_THRESHOLD,
) -> RoutingComposition:
    """Compute the routing composition of one benchmark's decisions.

    Args:
        decisions: The coordinator's ``(model, role)`` picks (see
            :func:`_normalize_decisions` for accepted shapes).
        benchmark: Name for the report row.
        pool_models: The full model pool. Used only to report models that were in
            the pool but never routed to; when ``None`` only observed models appear.
        collapse_threshold: A model's share at or above this flags ``collapsed``.

    Returns:
        A :class:`RoutingComposition`.
    """
    pairs = _normalize_decisions(decisions)
    n = len(pairs)
    model_counts = Counter(m for m, _ in pairs)
    role_counts = Counter(r for _, r in pairs if r is not None)

    model_shares = _shares(model_counts, n)
    role_total = sum(role_counts.values())
    role_shares = _shares(role_counts, role_total)

    observed = set(model_counts)
    pool = list(pool_models) if pool_models is not None else []
    unused_models = [m for m in pool if m not in observed]
    unused_roles = [r for r in ROLE_NAMES if role_counts.get(r, 0) == 0]

    if model_counts:
        top_model, top_count = model_counts.most_common(1)[0]
        top_share = top_count / n if n else 0.0
    else:
        top_model, top_share = None, 0.0

    collapsed = n > 0 and top_share >= collapse_threshold
    return RoutingComposition(
        benchmark=benchmark,
        n_decisions=n,
        model_shares=model_shares,
        role_shares=role_shares,
        unused_models=unused_models,
        unused_roles=unused_roles,
        model_entropy=_normalized_entropy(model_counts.values()),
        top_model=top_model,
        top_model_share=top_share,
        collapsed=collapsed,
    )


def analyze_benchmarks(
    per_benchmark: Mapping[str, Any],
    *,
    pool_models: Sequence[str] | None = None,
    collapse_threshold: float = _DEFAULT_COLLAPSE_THRESHOLD,
) -> dict[str, Any]:
    """Per-benchmark routing composition plus the pooled-union composition.

    Args:
        per_benchmark: ``{benchmark: decisions}``.
        pool_models: The full model pool (see :func:`analyze`).
        collapse_threshold: Collapse flag threshold (see :func:`analyze`).

    Returns:
        ``{"per_benchmark": [RoutingComposition.to_dict, ...], "union": <the same
        over all decisions pooled>, "any_collapsed": bool, "collapsed_benchmarks":
        [name, ...]}``.
    """
    results = [
        analyze(dec, benchmark=str(b), pool_models=pool_models,
                collapse_threshold=collapse_threshold)
        for b, dec in sorted(per_benchmark.items())
    ]
    pooled: list[Any] = []
    for dec in per_benchmark.values():
        pooled.extend(dec)
    union = analyze(pooled, benchmark="union", pool_models=pool_models,
                    collapse_threshold=collapse_threshold)
    collapsed_benchmarks = [r.benchmark for r in results if r.collapsed]
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "union": union.to_dict(),
        "any_collapsed": bool(collapsed_benchmarks),
        "collapsed_benchmarks": collapsed_benchmarks,
    }


def _fmt_shares(shares: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in
                     sorted(shares.items(), key=lambda kv: kv[1], reverse=True)) or "-"


def render(
    per_benchmark: Mapping[str, Any],
    *,
    pool_models: Sequence[str] | None = None,
    collapse_threshold: float = _DEFAULT_COLLAPSE_THRESHOLD,
) -> str:
    """A compact text report of the per-benchmark routing composition."""
    report = analyze_benchmarks(per_benchmark, pool_models=pool_models,
                                collapse_threshold=collapse_threshold)
    lines = ["| benchmark | n | model shares | role shares | entropy | collapsed |",
             "|---|---|---|---|---|---|"]
    for r in report["per_benchmark"] + [report["union"]]:
        lines.append(
            f"| {r['benchmark']} | {r['n_decisions']} | {_fmt_shares(r['model_shares'])} | "
            f"{_fmt_shares(r['role_shares'])} | {r['model_entropy']:.2f} | "
            f"{'YES' if r['collapsed'] else 'no'} |"
        )
    lines.append("")
    if report["any_collapsed"]:
        lines.append("collapsed (one model >= threshold) on: "
                     + ", ".join(report["collapsed_benchmarks"]))
    unused = report["union"]["unused_roles"]
    if unused:
        lines.append(f"roles never used (union): {', '.join(unused)}")
    return "\n".join(lines)
