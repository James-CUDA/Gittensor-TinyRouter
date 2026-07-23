"""Reconcile a fugu cost *projection* against the *actual* metered spend.

``fugu.cost.estimate_grpo_cost`` / ``estimate_eval_cost`` produce a pre-run projection
(``total_usd``, ``worker_calls``, ``conductor_calls``, plus the token assumptions), and
``fugu.cost.CostMeter.report`` records the actual spend (``spend_usd``, ``llm_calls``,
``prompt_tokens``, ``completion_tokens``). But nothing ever closes the loop — the two
estimators have no consumer, so no one asks *"did the run cost what we projected?"*

This lines the projection up against a ``CostMeter.report()``-shaped actual block and
reports, per field (spend, total LLM calls, and — when the projection carries token
assumptions — prompt/completion tokens): the absolute delta, the ``actual/projected``
ratio, and an over/under-run flag against a tolerance. Projected tokens are reconstructed
from the projection's own ``worker_calls``/``conductor_calls`` × its ``assumptions`` averages
(the only place the projected token counts live). Degrades to zeros on missing fields and
never raises. Pure data/number — no torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "FieldReconciliation",
    "CostReconciliation",
    "reconcile_projection",
    "render",
]

_DEFAULT_TOL = 0.15


def _num(x: Any) -> float:
    """Coerce to float; non-numbers -> 0.0 (mirrors ``cost_audit._num``, never raises)."""
    if isinstance(x, bool) or x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class FieldReconciliation:
    """Projected vs actual for one cost dimension."""

    field: str
    projected: float
    actual: float
    delta: float                 # actual - projected
    ratio: float | None          # actual / projected (None when projected == 0)
    overrun: bool                # actual exceeds projected by more than the tolerance
    underrun: bool               # actual is below projected by more than the tolerance

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "field": self.field,
            "projected": self.projected,
            "actual": self.actual,
            "delta": self.delta,
            "ratio": self.ratio,
            "overrun": self.overrun,
            "underrun": self.underrun,
        }


@dataclass(frozen=True)
class CostReconciliation:
    """Projection-vs-actual reconciliation across every comparable cost field."""

    tolerance: float
    fields: list[FieldReconciliation]
    within_tolerance: bool       # no field over- or under-ran beyond the tolerance

    def field(self, name: str) -> FieldReconciliation | None:
        """The reconciliation for ``name``, or None if that field wasn't compared."""
        return next((f for f in self.fields if f.field == name), None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
            "fields": [f.to_dict() for f in self.fields],
        }


def _reconcile_field(name: str, projected: float, actual: float, tol: float) -> FieldReconciliation:
    delta = actual - projected
    ratio = actual / projected if abs(projected) > 0 else None
    overrun = actual > projected * (1.0 + tol)
    underrun = actual < projected * (1.0 - tol)
    return FieldReconciliation(name, projected, actual, delta, ratio, overrun, underrun)


def _projected_tokens(estimate: Mapping[str, Any]) -> tuple[float, float]:
    """Reconstruct projected (prompt, completion) tokens from the projection's assumptions.

    ``worker_calls * avg_worker_tokens + conductor_calls * avg_conductor_tokens`` — the
    projection does not emit token totals directly, but it carries the per-call averages
    it used. Returns ``(0.0, 0.0)`` when the assumptions are absent.
    """
    assumptions = estimate.get("assumptions")
    if not isinstance(assumptions, Mapping):
        return 0.0, 0.0
    wt = assumptions.get("avg_worker_tokens") or [0, 0]
    ct = assumptions.get("avg_conductor_tokens") or [0, 0]
    if not (isinstance(wt, (list, tuple)) and isinstance(ct, (list, tuple))
            and len(wt) >= 2 and len(ct) >= 2):
        return 0.0, 0.0
    wc = _num(estimate.get("worker_calls"))
    cc = _num(estimate.get("conductor_calls"))
    prompt = wc * _num(wt[0]) + cc * _num(ct[0])
    completion = wc * _num(wt[1]) + cc * _num(ct[1])
    return prompt, completion


def reconcile_projection(
    estimate: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    tol: float = _DEFAULT_TOL,
) -> CostReconciliation:
    """Reconcile a cost projection against a ``CostMeter.report()``-shaped actual block.

    Args:
        estimate: an ``estimate_grpo_cost`` / ``estimate_eval_cost`` projection dict.
        actual: a ``CostMeter.report()`` dict (``spend_usd`` / ``llm_calls`` /
            ``prompt_tokens`` / ``completion_tokens``).
        tol: fractional tolerance; a field flags ``overrun`` when the actual exceeds the
            projection by more than ``tol`` (and ``underrun`` symmetrically).

    Returns:
        A :class:`CostReconciliation`. ``spend_usd`` and total ``llm_calls`` are always
        compared; token fields are added only when the projection carries token
        assumptions to reconstruct a non-zero projected count.
    """
    fields: list[FieldReconciliation] = []
    fields.append(_reconcile_field(
        "spend_usd", _num(estimate.get("total_usd")), _num(actual.get("spend_usd")), tol))
    projected_calls = _num(estimate.get("worker_calls")) + _num(estimate.get("conductor_calls"))
    fields.append(_reconcile_field(
        "llm_calls", projected_calls, _num(actual.get("llm_calls")), tol))

    proj_prompt, proj_completion = _projected_tokens(estimate)
    if proj_prompt > 0:
        fields.append(_reconcile_field(
            "prompt_tokens", proj_prompt, _num(actual.get("prompt_tokens")), tol))
    if proj_completion > 0:
        fields.append(_reconcile_field(
            "completion_tokens", proj_completion, _num(actual.get("completion_tokens")), tol))

    within = not any(f.overrun or f.underrun for f in fields)
    return CostReconciliation(tolerance=tol, fields=fields, within_tolerance=within)


def render(rec: CostReconciliation) -> str:
    """Markdown: the per-field projected/actual table + the overall verdict."""
    out = ["# Fugu cost reconciliation (projected vs actual)\n"]
    out.append("| field | projected | actual | delta | ratio | flag |")
    out.append("|---|---|---|---|---|---|")
    for f in rec.fields:
        ratio = f"{f.ratio:.2f}x" if f.ratio is not None else "—"
        flag = "over" if f.overrun else ("under" if f.underrun else "ok")
        out.append(f"| {f.field} | {f.projected:,.2f} | {f.actual:,.2f} | {f.delta:+,.2f} "
                   f"| {ratio} | {flag} |")
    verdict = ("within tolerance" if rec.within_tolerance
               else "OUT of tolerance — actual diverged from the projection")
    out.append(f"\n**Verdict** (±{rec.tolerance:.0%}): {verdict}.")
    return "\n".join(out) + "\n"
