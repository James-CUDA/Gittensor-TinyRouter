"""Offline Milestone-1 triage metrics (domain + difficulty)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TriageMetrics:
    """Aggregated M1 scores on a held-out set."""

    n: int
    domain_accuracy: float
    domain_macro_f1: float
    difficulty_exact: float
    difficulty_within_1: float
    joint_accuracy: float
    composite: float  # 0.7 * domain_acc + 0.3 * diff_exact
    per_domain_accuracy: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1s.append(_safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0)
    return float(sum(f1s) / len(f1s))


def score_triage(
    y_domain: Sequence[str],
    y_diff: Sequence[int],
    pred_domain: Sequence[str],
    pred_diff: Sequence[int],
    *,
    domain_weight: float = 0.7,
) -> TriageMetrics:
    """Compute M1 metrics. Lengths must match."""
    n = len(y_domain)
    if not (n == len(y_diff) == len(pred_domain) == len(pred_diff)):
        raise ValueError("all label/prediction sequences must have the same length")
    if n == 0:
        return TriageMetrics(
            n=0,
            domain_accuracy=0.0,
            domain_macro_f1=0.0,
            difficulty_exact=0.0,
            difficulty_within_1=0.0,
            joint_accuracy=0.0,
            composite=0.0,
            per_domain_accuracy={},
        )

    dom_ok = [t == p for t, p in zip(y_domain, pred_domain)]
    diff_ok = [int(t) == int(p) for t, p in zip(y_diff, pred_diff)]
    within1 = [abs(int(t) - int(p)) <= 1 for t, p in zip(y_diff, pred_diff)]
    joint = [a and b for a, b in zip(dom_ok, diff_ok)]

    # per-domain accuracy
    tot: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    for t, ok in zip(y_domain, dom_ok):
        tot[t] += 1
        if ok:
            hit[t] += 1
    per_dom = {d: _safe_div(hit[d], tot[d]) for d in sorted(tot)}

    d_acc = _safe_div(sum(dom_ok), n)
    f_exact = _safe_div(sum(diff_ok), n)
    w_diff = float(domain_weight)
    composite = w_diff * d_acc + (1.0 - w_diff) * f_exact

    return TriageMetrics(
        n=n,
        domain_accuracy=d_acc,
        domain_macro_f1=macro_f1(list(y_domain), list(pred_domain)),
        difficulty_exact=f_exact,
        difficulty_within_1=_safe_div(sum(within1), n),
        joint_accuracy=_safe_div(sum(joint), n),
        composite=composite,
        per_domain_accuracy=per_dom,
    )
