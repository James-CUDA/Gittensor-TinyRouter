"""Offline R5 check: does TRINITY reach the Per-Question-Best ceiling?

``docs/SPEC.md`` §1.3 invariant **R5** — *"TRINITY ≈ Per-Question-Best on 3 of 4
in-dist tasks"* — is a replication requirement, yet nothing in ``src/`` or
``scripts/`` verifies it. The **Per-Question-Best (PQB)** upper bound is the
*"union of correct answers across the pool"* (SPEC §5.2): the fraction of
questions some pool model answers correctly, i.e. the accuracy a per-question
oracle router would reach. R5 is the claim that the *trained* router very nearly
reaches that ceiling — it captures the routable signal, not just part of it.

This reads, per benchmark, TRINITY's accuracy and the benchmark's PQB ceiling,
and reports the gap ``PQB - TRINITY``, whether TRINITY is within ``tol`` of the
ceiling ("≈"), and the R5 verdict across benchmarks (TRINITY is ≈ PQB on at least
``min_pass`` of them — by default all-but-one, matching the paper's "3 of 4").
Equal benchmark weighting for the union mean gap, matching the composite score
(ROADMAP Phase 2).

``per_question_best`` is the same quantity ``trinity.analysis.union_oracle`` and
``scripts/oracle_ceiling.py`` already estimate (the routing/union oracle) — this
module does not recompute it; it consumes it and adds the TRINITY comparison R5
needs. Pure numpy/stdlib over plain numbers — no torch, no network, no GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "PQBGap",
    "analyze_task",
    "analyze_benchmarks",
    "render",
]

# Default "≈" tolerance: TRINITY counts as reaching the ceiling when it is within
# 5 accuracy points of the Per-Question-Best oracle. Configurable per call.
_DEFAULT_TOL = 0.05


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class PQBGap:
    """R5 diagnostics for one benchmark: TRINITY vs the Per-Question-Best ceiling.

    ``gap`` is ``per_question_best - trinity_accuracy`` — how far TRINITY sits
    below the per-question oracle (negative if TRINITY meets or exceeds it, which a
    noisy estimate can produce). ``reaches`` is the R5 "≈" test for this benchmark:
    ``gap <= tol``. ``fraction_closed`` is the share of the routable headroom above
    a random pick that TRINITY captured (``NaN`` when there is no headroom to close).
    """

    benchmark: str
    trinity_accuracy: float
    per_question_best: float
    gap: float
    reaches: bool
    fraction_closed: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "trinity_accuracy": self.trinity_accuracy,
            "per_question_best": self.per_question_best,
            "gap": self.gap,
            "reaches": self.reaches,
            "fraction_closed": self.fraction_closed,
        }


def analyze_task(
    benchmark: str,
    trinity_accuracy: float,
    per_question_best: float,
    *,
    tol: float = _DEFAULT_TOL,
    baseline: float | None = None,
) -> PQBGap:
    """Compute the R5 gap-to-ceiling for one benchmark.

    Args:
        benchmark: Name for the report row.
        trinity_accuracy: The trained router's accuracy on this benchmark, in
            ``[0, 1]``.
        per_question_best: The Per-Question-Best (union-of-correct) ceiling for
            this benchmark, in ``[0, 1]`` — e.g. ``routing_oracle`` from
            :mod:`trinity.analysis.union_oracle`.
        tol: "≈" tolerance. TRINITY ``reaches`` the ceiling iff
            ``per_question_best - trinity_accuracy <= tol``.
        baseline: Optional reference accuracy (e.g. random routing or best single
            model) used only for ``fraction_closed`` — the share of the headroom
            ``per_question_best - baseline`` that TRINITY captured. When omitted or
            when there is no headroom, ``fraction_closed`` is ``NaN``.

    Returns:
        A :class:`PQBGap`.
    """
    ta = float(trinity_accuracy)
    pqb = float(per_question_best)
    gap = pqb - ta
    reaches = gap <= tol
    fraction_closed = math.nan
    if baseline is not None and _is_num(baseline):
        headroom = pqb - float(baseline)
        if headroom > tol:
            fraction_closed = (ta - float(baseline)) / headroom
    return PQBGap(
        benchmark=benchmark,
        trinity_accuracy=ta,
        per_question_best=pqb,
        gap=gap,
        reaches=reaches,
        fraction_closed=fraction_closed,
    )


def analyze_benchmarks(
    tasks: Mapping[str, Any],
    *,
    tol: float = _DEFAULT_TOL,
    min_pass: int | None = None,
) -> dict[str, Any]:
    """Per-benchmark R5 gaps plus the cross-benchmark verdict.

    Args:
        tasks: ``{benchmark: entry}`` where ``entry`` is either the raw
            ``per_question_best`` number or a mapping carrying ``trinity`` /
            ``trinity_accuracy`` and ``per_question_best`` / ``pqb`` (and an
            optional ``baseline``). A benchmark whose TRINITY or PQB value is
            missing/non-numeric is skipped.
        tol: "≈" tolerance (see :func:`analyze_task`).
        min_pass: How many benchmarks must reach the ceiling for R5 to hold.
            Defaults to *all but one* (``n - 1``, min 1), matching the paper's
            "3 of 4 in-dist tasks".

    Returns:
        ``{"per_benchmark": [PQBGap.to_dict, ...], "n_reached": int,
           "n_scored": int, "min_pass": int, "r5_holds": bool,
           "union_mean_gap": float, "misses": [benchmark, ...]}``.
    """
    results: list[PQBGap] = []
    for bench, entry in sorted(tasks.items()):
        if isinstance(entry, Mapping):
            ta = entry.get("trinity", entry.get("trinity_accuracy"))
            pqb = entry.get("per_question_best", entry.get("pqb"))
            baseline = entry.get("baseline")
        elif _is_num(entry):
            ta, pqb, baseline = None, entry, None
        else:
            continue
        if not (_is_num(ta) and _is_num(pqb)):
            continue
        results.append(
            analyze_task(str(bench), float(ta), float(pqb), tol=tol,
                         baseline=baseline if _is_num(baseline) else None)
        )

    n_scored = len(results)
    n_reached = sum(r.reaches for r in results)
    if min_pass is None:
        min_pass = max(1, n_scored - 1)
    union_mean_gap = (sum(r.gap for r in results) / n_scored) if n_scored else 0.0
    misses = [r.benchmark for r in results if not r.reaches]
    return {
        "per_benchmark": [r.to_dict() for r in results],
        "n_reached": n_reached,
        "n_scored": n_scored,
        "min_pass": min_pass,
        "r5_holds": n_scored > 0 and n_reached >= min_pass,
        "union_mean_gap": union_mean_gap,
        "misses": misses,
    }


def render(
    tasks: Mapping[str, Any], *, tol: float = _DEFAULT_TOL, min_pass: int | None = None,
) -> str:
    """A compact text report of the per-benchmark R5 check and the union verdict."""
    report = analyze_benchmarks(tasks, tol=tol, min_pass=min_pass)
    lines = ["| benchmark | TRINITY | per-question-best | gap | ≈ |",
             "|---|---|---|---|---|"]
    for r in report["per_benchmark"]:
        flag = "ok" if r["reaches"] else f"short {r['gap']:+.3f}"
        lines.append(
            f"| {r['benchmark']} | {r['trinity_accuracy']:.3f} | "
            f"{r['per_question_best']:.3f} | {r['gap']:+.3f} | {flag} |"
        )
    verdict = "HOLDS" if report["r5_holds"] else "VIOLATED"
    lines.append("")
    lines.append(
        f"R5 (TRINITY ≈ Per-Question-Best): {verdict} — reached the ceiling on "
        f"{report['n_reached']}/{report['n_scored']} benchmarks "
        f"(need {report['min_pass']}; union mean gap {report['union_mean_gap']:+.3f})"
    )
    if report["misses"]:
        lines.append(f"short of ceiling: {', '.join(report['misses'])}")
    return "\n".join(lines)
