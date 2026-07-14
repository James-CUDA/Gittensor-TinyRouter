"""Offline R7 check: does accuracy rise monotonically as the turn budget grows?

``docs/SPEC.md`` §1.3 invariant **R7** — *"More max-turns -> monotonic gain"*
(target 0.823 -> 0.863 over 2 -> 6 turns) — is a replication requirement, yet
nothing in ``src/`` or ``scripts/`` verifies it. R7 is what justifies the
multi-turn loop at all: re-evaluating the SAME trained head with a larger
``max_turns`` budget should never lower accuracy and should net a gain; a drop
means extra turns are hurting (a Verifier over-REVISING, a Worker talking itself
out of a right answer), which the loop is meant to avoid.

This reads a turn sweep -- the accuracy of one head evaluated at several
``max_turns`` values on a benchmark -- and reports, per benchmark and for the
3-benchmark union, whether accuracy is monotone non-decreasing in the budget, the
net gain from the smallest to the largest budget, and the worst downward step (an
R7 violation). Equal benchmark weighting for the union, matching the composite
score (ROADMAP Phase 2).

Pure numpy/stdlib over plain numbers -- no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "TurnsSweep",
    "analyze_sweep",
    "analyze_benchmarks",
    "render",
]

_TOL = 1e-9


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _clean_points(points: Any) -> list[tuple[int, float]]:
    """Coerce a turns->accuracy mapping (or (turns, acc) pairs) to sorted points.

    Accepts a ``{max_turns: accuracy}`` mapping or an iterable of ``(max_turns,
    accuracy)`` pairs. Non-numeric or duplicate-turn entries are dropped (the last
    value wins for a duplicate), and points are returned sorted by ``max_turns``.
    """
    by_turns: dict[int, float] = {}
    items = points.items() if isinstance(points, Mapping) else points
    for entry in items:
        try:
            t, a = entry
        except (TypeError, ValueError):
            continue
        if _is_num(t) and _is_num(a):
            by_turns[int(t)] = float(a)
    return sorted(by_turns.items())


@dataclass(frozen=True)
class TurnsSweep:
    """R7 diagnostics for one benchmark's ``max_turns`` -> accuracy sweep.

    ``monotone`` is True iff accuracy never falls (within ``tol``) as the budget
    grows. ``net_gain`` is ``accuracy[max budget] - accuracy[min budget]``.
    ``max_drop`` is the largest single-step decrease (0.0 when monotone), and
    ``worst_step`` names the ``(from_turns, to_turns)`` where it occurred.
    ``holds`` is the R7 verdict: monotone AND a positive net gain.
    """

    benchmark: str
    turns: list[int]
    accuracies: list[float]
    monotone: bool
    net_gain: float
    max_drop: float
    worst_step: tuple[int, int] | None
    holds: bool

    @property
    def n_points(self) -> int:
        """Number of distinct turn budgets in the sweep."""
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "turns": list(self.turns),
            "accuracies": list(self.accuracies),
            "n_points": self.n_points,
            "monotone": self.monotone,
            "net_gain": self.net_gain,
            "max_drop": self.max_drop,
            "worst_step": list(self.worst_step) if self.worst_step else None,
            "holds": self.holds,
        }


def analyze_sweep(
    points: Any, *, benchmark: str = "?", tol: float = _TOL,
) -> TurnsSweep:
    """Compute the R7 monotonicity diagnostics for one turn sweep.

    Args:
        points: ``{max_turns: accuracy}`` or ``(max_turns, accuracy)`` pairs. At
            least two distinct turn budgets are needed to judge a trend; a sweep
            with fewer is reported as non-holding with ``net_gain`` 0.
        benchmark: Name for the report row.
        tol: A step is a "drop" only if it falls by more than ``tol`` (so float
            noise on a flat step is not a violation).

    Returns:
        A :class:`TurnsSweep`.
    """
    pts = _clean_points(points)
    turns = [t for t, _ in pts]
    accs = [a for _, a in pts]
    if len(pts) < 2:
        return TurnsSweep(benchmark, turns, accs, False, 0.0, 0.0, None, False)

    max_drop = 0.0
    worst: tuple[int, int] | None = None
    for (t0, a0), (t1, a1) in zip(pts, pts[1:]):
        drop = a0 - a1
        if drop > max_drop:
            max_drop, worst = drop, (t0, t1)

    monotone = max_drop <= tol
    net_gain = accs[-1] - accs[0]
    holds = monotone and net_gain > tol
    return TurnsSweep(
        benchmark=benchmark, turns=turns, accuracies=accs,
        monotone=monotone, net_gain=net_gain,
        max_drop=max(0.0, max_drop), worst_step=worst, holds=holds,
    )


def analyze_benchmarks(sweeps: Mapping[str, Any], *, tol: float = _TOL) -> dict[str, Any]:
    """Per-benchmark R7 sweeps plus the equally-weighted union verdict.

    Args:
        sweeps: ``{benchmark: points}`` where each ``points`` is a turns->accuracy
            mapping or pair list.
        tol: Drop tolerance (see :func:`analyze_sweep`).

    Returns:
        ``{"per_benchmark": [TurnsSweep.to_dict, ...], "r7_holds": bool,
           "union_net_gain": float, "violations": [benchmark, ...]}``. ``r7_holds``
        is True iff every benchmark's sweep holds; ``union_net_gain`` is the
        equal-weight mean net gain across benchmarks.
    """
    results = [analyze_sweep(pts, benchmark=str(b), tol=tol)
               for b, pts in sorted(sweeps.items())]
    scored = [r for r in results if r.n_points >= 2]
    violations = [r.benchmark for r in results if not r.holds]
    union_net_gain = (sum(r.net_gain for r in scored) / len(scored)) if scored else 0.0
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "r7_holds": bool(scored) and all(r.holds for r in results),
        "union_net_gain": union_net_gain,
        "violations": violations,
    }


def render(sweeps: Mapping[str, Any], *, tol: float = _TOL) -> str:
    """A compact text report of the per-benchmark R7 check and the union verdict."""
    report = analyze_benchmarks(sweeps, tol=tol)
    lines = ["| benchmark | turns | accuracies | net gain | max drop | R7 |",
             "|---|---|---|---|---|---|"]
    for r in report["per_benchmark"]:
        turns = ",".join(str(t) for t in r["turns"]) or "-"
        accs = "->".join(f"{a:.3f}" for a in r["accuracies"]) or "-"
        flag = "ok" if r["holds"] else ("not monotone" if not r["monotone"] else "no gain")
        lines.append(
            f"| {r['benchmark']} | {turns} | {accs} | {r['net_gain']:+.3f} | "
            f"{r['max_drop']:.3f} | {flag} |"
        )
    verdict = "HOLDS" if report["r7_holds"] else "VIOLATED"
    lines.append("")
    lines.append(f"R7 (more max-turns -> monotonic gain): {verdict} "
                 f"(union mean net gain {report['union_net_gain']:+.3f})")
    if report["violations"]:
        lines.append(f"violations: {', '.join(report['violations'])}")
    return "\n".join(lines)
