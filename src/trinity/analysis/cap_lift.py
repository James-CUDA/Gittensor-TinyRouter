"""Offline R6 check: does lifting the token cap jump accuracy past every constituent?

``docs/SPEC.md`` §1.3 invariant **R6** — *"Lifting token cap -> large LiveCodeBench
jump, beats all constituents"* (0.61 -> 0.862, beats GPT-5 0.838) — is a replication
requirement, yet nothing in ``src/`` or ``scripts/`` verifies it. SPEC §7 spells out
the procedure: *"After training, lift the 4096-token cap, no retraining (R6) ...
simply remove the cap and re-eval."*

R6 is the headline routing claim on the reasoning-heavy split: the SAME trained
coordinator, re-evaluated with the per-turn output-token cap raised (4096 -> uncapped)
and **no retraining**, should (a) jump a lot and (b) land above *every* constituent
single model — GPT-5 included. It is what shows the win is real coordination headroom
the token cap was hiding, not a quirk of a stingy decode budget. A jump that still
trails the best constituent means the routed system only caught up, not overtook.

This is distinct from R7 (:mod:`trinity.analysis.turns_monotonicity`, the *max-turns*
axis, monotone gain only): R6 sweeps the *token-budget* axis and adds the
**beats-all-constituents** dominance test. Given a token-cap sweep of the routed
accuracy on a benchmark plus the constituent single-model accuracies at the lifted
cap, it reports the lift jump, whether accuracy is monotone non-decreasing in the cap
(lifting the budget should never hurt), the margin over the best constituent, and the
R6 verdict.

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "TokenCapLift",
    "analyze_cap_lift",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_cap(x: Any) -> float | None:
    """Coerce a cap key to a float; ``inf``/``uncapped``/``none``/None -> +inf.

    The lifted (removed) cap is naturally represented as ``+inf`` so it sorts last and
    is always the largest budget. Unparseable values return None (dropped by the caller).
    """
    if x is None:
        return math.inf
    if _is_num(x):
        return math.inf if math.isinf(float(x)) else float(x)
    s = str(x).strip().lower()
    if s in {"inf", "+inf", "infinity", "uncapped", "none", "null", "off", ""}:
        return math.inf
    try:
        return float(s)
    except ValueError:
        return None


def _clean_points(points: Any) -> list[tuple[float, float]]:
    """Coerce a cap->accuracy mapping (or ``(cap, accuracy)`` pairs) to sorted points.

    Accepts a ``{cap: accuracy}`` mapping or an iterable of ``(cap, accuracy)`` pairs.
    Non-numeric accuracies and unparseable caps are dropped; a duplicate cap keeps the
    last value. Points are returned sorted by cap ascending, with an uncapped (+inf)
    budget sorting last.
    """
    by_cap: dict[float, float] = {}
    items = points.items() if isinstance(points, Mapping) else points
    for entry in items:
        try:
            c, a = entry
        except (TypeError, ValueError):
            continue
        cap = _as_cap(c)
        if cap is not None and _is_num(a):
            by_cap[cap] = float(a)
    return sorted(by_cap.items())


def _clean_constituents(constituents: Any) -> dict[str, float]:
    """Coerce a ``{model: accuracy}`` mapping to a clean float dict (drop non-numeric)."""
    if not isinstance(constituents, Mapping):
        return {}
    return {str(m): float(a) for m, a in constituents.items() if _is_num(a)}


def _fmt_cap(cap: float) -> str:
    """Human label for a cap: ``uncapped`` for +inf, else the integer/float budget."""
    if math.isinf(cap):
        return "uncapped"
    return str(int(cap)) if float(cap).is_integer() else str(cap)


@dataclass(frozen=True)
class TokenCapLift:
    """R6 diagnostics for one benchmark's token-cap -> accuracy sweep.

    ``jump`` is ``accuracy[max cap] - accuracy[min cap]`` (the lift from raising the
    budget). ``monotone`` is True iff accuracy never falls (within ``tol``) as the cap
    grows. ``beats_all_constituents`` is True iff the lifted accuracy strictly exceeds
    every constituent single model; ``margin`` is the lifted accuracy minus the best
    constituent. ``holds`` is the R6 verdict: a positive jump AND beating every
    constituent (when constituents are supplied).
    """

    benchmark: str
    caps: list[float]
    accuracies: list[float]
    base_cap: float
    base_accuracy: float
    lifted_cap: float
    lifted_accuracy: float
    jump: float
    monotone: bool
    max_drop: float
    worst_step: tuple[float, float] | None
    constituents: dict[str, float]
    best_constituent: str | None
    best_constituent_acc: float
    beats_all_constituents: bool
    margin: float
    holds: bool

    @property
    def n_points(self) -> int:
        """Number of distinct token-cap levels in the sweep."""
        return len(self.caps)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (caps rendered as labels; ``uncapped`` for +inf)."""
        return {
            "benchmark": self.benchmark,
            "caps": [_fmt_cap(c) for c in self.caps],
            "accuracies": list(self.accuracies),
            "n_points": self.n_points,
            "base_cap": _fmt_cap(self.base_cap),
            "base_accuracy": self.base_accuracy,
            "lifted_cap": _fmt_cap(self.lifted_cap),
            "lifted_accuracy": self.lifted_accuracy,
            "jump": self.jump,
            "monotone": self.monotone,
            "max_drop": self.max_drop,
            "worst_step": (
                [_fmt_cap(self.worst_step[0]), _fmt_cap(self.worst_step[1])]
                if self.worst_step else None
            ),
            "constituents": dict(self.constituents),
            "best_constituent": self.best_constituent,
            "best_constituent_acc": self.best_constituent_acc,
            "beats_all_constituents": self.beats_all_constituents,
            "margin": self.margin,
            "holds": self.holds,
        }


def analyze_cap_lift(
    points: Any,
    constituents: Any = None,
    *,
    benchmark: str = "?",
    tol: float = _TOL,
) -> TokenCapLift:
    """Compute the R6 token-cap-lift diagnostics for one benchmark.

    Args:
        points: ``{cap: accuracy}`` or ``(cap, accuracy)`` pairs. A cap may be a number
            or ``inf``/``uncapped``/None for the lifted (removed) budget. At least two
            distinct caps are needed to measure a lift; with fewer the sweep is reported
            as non-holding with ``jump`` 0.
        constituents: optional ``{model: accuracy}`` of the constituent single models at
            the lifted cap. When omitted, ``beats_all_constituents`` is False and the R6
            verdict cannot hold (the dominance half of R6 is unproven).
        benchmark: name for the report row (R6 is the LiveCodeBench claim).
        tol: a step counts as a "drop" only if it falls by more than ``tol``, and the
            lifted accuracy must exceed a constituent by more than ``tol`` to beat it.

    Returns:
        A :class:`TokenCapLift`.
    """
    pts = _clean_points(points)
    caps = [c for c, _ in pts]
    accs = [a for _, a in pts]
    cons = _clean_constituents(constituents)

    best_model: str | None = None
    best_acc = 0.0
    if cons:
        best_model = max(cons, key=lambda m: cons[m])
        best_acc = cons[best_model]

    if len(pts) < 2:
        base_cap = caps[0] if caps else math.inf
        base_acc = accs[0] if accs else 0.0
        return TokenCapLift(
            benchmark=benchmark, caps=caps, accuracies=accs,
            base_cap=base_cap, base_accuracy=base_acc,
            lifted_cap=base_cap, lifted_accuracy=base_acc,
            jump=0.0, monotone=len(pts) == 1, max_drop=0.0, worst_step=None,
            constituents=cons, best_constituent=best_model, best_constituent_acc=best_acc,
            beats_all_constituents=False, margin=base_acc - best_acc, holds=False,
        )

    max_drop = 0.0
    worst: tuple[float, float] | None = None
    for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
        drop = a0 - a1
        if drop > max_drop:
            max_drop, worst = drop, (c0, c1)

    monotone = max_drop <= tol
    lifted_acc = accs[-1]
    jump = lifted_acc - accs[0]
    beats_all = bool(cons) and all(lifted_acc > a + tol for a in cons.values())
    holds = jump > tol and beats_all
    return TokenCapLift(
        benchmark=benchmark, caps=caps, accuracies=accs,
        base_cap=caps[0], base_accuracy=accs[0],
        lifted_cap=caps[-1], lifted_accuracy=lifted_acc,
        jump=jump, monotone=monotone, max_drop=max(0.0, max_drop), worst_step=worst,
        constituents=cons, best_constituent=best_model, best_constituent_acc=best_acc,
        beats_all_constituents=beats_all, margin=lifted_acc - best_acc, holds=holds,
    )


def render(
    points: Any,
    constituents: Any = None,
    *,
    benchmark: str = "livecodebench",
    tol: float = _TOL,
) -> str:
    """A compact text report of the R6 token-cap-lift check and the verdict."""
    r = analyze_cap_lift(points, constituents, benchmark=benchmark, tol=tol)
    sweep = " -> ".join(
        f"{_fmt_cap(c)}:{a:.3f}" for c, a in zip(r.caps, r.accuracies)
    ) or "-"
    lines = [
        f"# R6 token-cap lift — {r.benchmark}\n",
        f"- cap sweep: {sweep}",
        f"- lift jump ({_fmt_cap(r.base_cap)} -> {_fmt_cap(r.lifted_cap)}): {r.jump:+.3f}",
        f"- monotone in cap (lifting never hurts): {r.monotone}",
    ]
    if r.constituents:
        cons = ", ".join(f"{m} {a:.3f}" for m, a in sorted(
            r.constituents.items(), key=lambda kv: -kv[1]))
        lines.append(f"- constituents: {cons}")
        lines.append(
            f"- lifted {r.lifted_accuracy:.3f} vs best constituent "
            f"{r.best_constituent} {r.best_constituent_acc:.3f}: margin {r.margin:+.3f}"
        )
    else:
        lines.append("- constituents: _(none supplied — dominance unproven)_")
    verdict = "HOLDS" if r.holds else "NOT SHOWN"
    lines.append(
        f"\n**R6** (lifting the token cap jumps past every constituent): {verdict}."
    )
    return "\n".join(lines) + "\n"
