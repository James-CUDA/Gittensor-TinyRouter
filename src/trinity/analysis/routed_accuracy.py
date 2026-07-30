"""Offline routed-accuracy diagnostic: is the head routing questions to the right model?

``trinity.analysis.sampling`` measures each model's *intrinsic* solve rate over ALL
questions (``p_hat`` from the K-sample oracle matrix). But that is not what the
trained coordinator's accuracy depends on: what matters is a model's accuracy on
exactly the questions the head **chose to route to it**. A model with a high
intrinsic solve rate but low *routed* accuracy means the head is sending it the
wrong questions — a routing problem the intrinsic rate cannot show, and the
aggregate ``routing_composition`` share (which counts picks, not correctness) does
not either.

This reads a run's ``(routed_model, correct)`` outcomes — the model the head routed
the answering turn to, and whether the trajectory was ultimately correct — and
reports, per model: how many questions it was routed (its **share**), its **routed
accuracy** on them, and its **contribution** to overall accuracy (share x accuracy).
It flags the model the head most **over-uses relative to its routed accuracy** (high
share, below-average routed accuracy) — the first place to re-balance routing.

Pure stdlib over plain ``(model, correct)`` records -- no torch, no network, no GPU.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = [
    "ModelRoutedAccuracy",
    "RoutedAccuracySummary",
    "analyze",
    "render",
]


def _as_bool(x: Any) -> bool | None:
    """Coerce a correctness value: bool, 0/1, or a score (``> 0`` => correct)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x > 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "correct", "1", "pass", "passed"}:
            return True
        if s in {"false", "incorrect", "wrong", "0", "fail", "failed"}:
            return False
    return None


def _record(rec: Any) -> tuple[str, bool] | None:
    """Coerce one outcome to ``(model, correct)``.

    Accepts a mapping with ``model`` and ``correct`` (aliases ``is_correct`` /
    ``score``), or a ``(model, correct)`` pair. Returns ``None`` when the model is
    empty or correctness is unusable.
    """
    model: Any
    correct: Any
    if isinstance(rec, Mapping):
        model = rec.get("model")
        correct = rec.get("correct", rec.get("is_correct", rec.get("score")))
    elif isinstance(rec, (tuple, list)) and len(rec) >= 2:
        model, correct = rec[0], rec[1]
    else:
        return None
    if model is None or model == "":
        return None
    c = _as_bool(correct)
    if c is None:
        return None
    return str(model), c


@dataclass(frozen=True)
class ModelRoutedAccuracy:
    """One model's routed slice: how often it was routed and its accuracy there."""

    model: str
    n_routed: int
    share: float          # fraction of all routed questions sent to this model
    routed_accuracy: float
    contribution: float   # share * routed_accuracy (adds up to overall accuracy)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "n_routed": self.n_routed,
            "share": self.share,
            "routed_accuracy": self.routed_accuracy,
            "contribution": self.contribution,
        }


@dataclass(frozen=True)
class RoutedAccuracySummary:
    """Per-model routed-accuracy breakdown for one run."""

    n_decisions: int
    overall_accuracy: float
    per_model: list[ModelRoutedAccuracy]   # sorted by share desc
    best_model: str | None                  # highest routed accuracy (>= 1 routed)
    worst_model: str | None                 # lowest routed accuracy
    overused_model: str | None              # largest share among below-average models

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "n_decisions": self.n_decisions,
            "overall_accuracy": self.overall_accuracy,
            "per_model": [m.to_dict() for m in self.per_model],
            "best_model": self.best_model,
            "worst_model": self.worst_model,
            "overused_model": self.overused_model,
        }


def analyze(records: Iterable[Any]) -> RoutedAccuracySummary:
    """Compute the per-model routed-accuracy breakdown.

    Args:
        records: ``(routed_model, correct)`` outcomes (see :func:`_record`). ``correct``
            may be a bool, a 0/1, or a score (``> 0`` counts as correct).

    Returns:
        A :class:`RoutedAccuracySummary`. Empty / all-unusable input yields a zeroed
        summary with ``None`` model fields.
    """
    n_by: dict[str, int] = defaultdict(int)
    correct_by: dict[str, int] = defaultdict(int)
    for rec in records:
        parsed = _record(rec)
        if parsed is None:
            continue
        model, correct = parsed
        n_by[model] += 1
        if correct:
            correct_by[model] += 1

    n_total = sum(n_by.values())
    if n_total == 0:
        return RoutedAccuracySummary(0, 0.0, [], None, None, None)

    rows = [
        ModelRoutedAccuracy(
            model=m,
            n_routed=n_by[m],
            share=n_by[m] / n_total,
            routed_accuracy=correct_by[m] / n_by[m],
            contribution=correct_by[m] / n_total,
        )
        for m in n_by
    ]
    rows.sort(key=lambda r: r.share, reverse=True)

    overall = sum(correct_by.values()) / n_total
    best = max(rows, key=lambda r: r.routed_accuracy).model
    worst = min(rows, key=lambda r: r.routed_accuracy).model
    # "over-used": routed below the overall accuracy, and among those the biggest share.
    below = [r for r in rows if r.routed_accuracy < overall]
    overused = max(below, key=lambda r: r.share).model if below else None
    return RoutedAccuracySummary(
        n_decisions=n_total,
        overall_accuracy=overall,
        per_model=rows,
        best_model=best,
        worst_model=worst,
        overused_model=overused,
    )


def render(records: Iterable[Any]) -> str:
    """A compact text report of the per-model routed accuracy."""
    s = analyze(list(records))
    lines = ["| model | routed | share | routed acc | contribution |",
             "|---|---|---|---|---|"]
    for r in s.per_model:
        lines.append(
            f"| {r.model} | {r.n_routed} | {r.share:.2f} | {r.routed_accuracy:.3f} | "
            f"{r.contribution:.3f} |"
        )
    lines.append("")
    lines.append(f"overall routed accuracy: {s.overall_accuracy:.3f} "
                 f"over {s.n_decisions} routed questions")
    if s.overused_model is not None:
        lines.append(f"over-used relative to its routed accuracy: {s.overused_model} "
                     "(high share, below-average accuracy — re-balance here first)")
    return "\n".join(lines)
