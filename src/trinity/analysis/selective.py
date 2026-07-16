"""Per-model selective-prediction / risk-coverage over the K-sample oracle matrix.

Every other ``oracle_matrix_<bench>.json`` consumer answers *every* query. But the
self-consistency of a model's K samples is a cheap confidence signal: when the samples
agree (``p_hat`` near 0 or 1) the majority answer is far more likely right than when they
split (``p_hat`` near 0.5). This module asks the selective-prediction question that
``the improvement plan`` #7 (the UCCI "confidence-based turn stopping" cascade) needs an
offline substrate for: **if a model abstains on its least self-consistent queries, how
much does accuracy on the answered set rise, and is that self-consistency confidence
actually informative?**

For each model it builds the risk-coverage curve — sort queries by self-consistency
confidence ``max(p_hat, 1 - p_hat)`` descending, and at each coverage report the accuracy
of the majority-vote answer on the answered set — and reports AURC (area under the risk
curve), accuracy at a few coverage levels, and the abstention gain.

The load-bearing, non-circular metric is **AURC vs a random-ordering baseline**: random
coverage has constant selective risk equal to the base error rate, so ``aurc_gain =
random_aurc - aurc > 0`` iff the confidence signal genuinely ranks likely-correct queries
ahead of likely-wrong ones. (Confidence and correctness are read from the same K samples —
the definition of *self*-consistency selective prediction; a fully leakage-free estimate
would use disjoint samples, but K is 3-5 here, too small to split reliably. The
random-baseline comparison is what keeps the "is it useful?" verdict honest.)

Distinct from ``sampling`` (#239: pass@1 / pass@K / majority@K, no coverage axis) and
``ensemble`` (#238: cross-model plurality). Pure numpy over the on-disk 0/1 matrix — no
torch, no network, no GPU. Meaningful only for K > 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trinity.analysis.sampling import solve_counts

__all__ = [
    "ModelSelective",
    "SelectiveSummary",
    "risk_coverage",
    "analyze",
    "render",
]

DEFAULT_COVERAGES = (1.0, 0.8, 0.5)
_USEFUL_EPS = 1e-3      # aurc_gain above this -> confidence is informative


def risk_coverage(correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: ``(coverage, selective_accuracy)`` sorted by confidence desc.

    At coverage ``k/n`` the model answers its ``k`` most-confident queries; the selective
    accuracy is the mean correctness over those ``k``. Within an equal-confidence group the
    queries are exchangeable, so each member contributes the group's *mean* correctness —
    i.e. the **expected** curve under uniform random tie-breaking. This makes AURC
    order-independent (a degenerate all-equal-confidence signal yields a flat curve at the
    base accuracy, not an artifact of the input order).
    """
    correct = np.asarray(correct, dtype=float).ravel()
    confidence = np.asarray(confidence, dtype=float).ravel()
    n = correct.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(-confidence, kind="stable")
    conf_sorted = confidence[order]
    corr_sorted = correct[order]
    # Replace each tie group's members by the group mean == expected per-position
    # correctness under random tie-breaking (so cumsum is the expected cumulative correct).
    expected = corr_sorted.copy()
    start = 0
    for i in range(1, n + 1):
        if i == n or conf_sorted[i] != conf_sorted[start]:
            expected[start:i] = corr_sorted[start:i].mean()
            start = i
    ks = np.arange(1, n + 1)
    selective_accuracy = np.cumsum(expected) / ks
    coverage = ks / n
    return coverage, selective_accuracy


def _accuracy_at(selective_accuracy: np.ndarray, cov: float) -> float:
    """Selective accuracy at coverage ``cov`` (answer the top ``round(cov*n)`` queries)."""
    n = selective_accuracy.shape[0]
    if n == 0:
        return 0.0
    idx = min(max(1, int(round(cov * n))), n)
    return float(selective_accuracy[idx - 1])


@dataclass(frozen=True)
class ModelSelective:
    """One model's selective-prediction profile from its self-consistency confidence."""

    model: str
    base_accuracy: float                    # majority@K accuracy at full coverage
    aurc: float                             # area under the selective RISK curve (lower better)
    random_aurc: float                      # AURC under random coverage == base error rate
    aurc_gain: float                        # random_aurc - aurc (>0 -> confidence informative)
    accuracy_at_coverage: dict[float, float]
    abstention_gain: float                  # acc@0.8 - acc@1.0 (does dropping low-conf help?)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (coverage keys stringified)."""
        return {
            "model": self.model,
            "base_accuracy": self.base_accuracy,
            "aurc": self.aurc,
            "random_aurc": self.random_aurc,
            "aurc_gain": self.aurc_gain,
            "accuracy_at_coverage": {f"{c:.2f}": a for c, a in self.accuracy_at_coverage.items()},
            "abstention_gain": self.abstention_gain,
        }


@dataclass(frozen=True)
class SelectiveSummary:
    """Per-model selective prediction + whether self-consistency confidence is informative."""

    benchmark: str
    n_questions: int
    k: int
    coverages: list[float]
    models: list[str]
    per_model: list[ModelSelective]
    best_aurc_model: str | None             # lowest AURC (best-calibrated selective risk)
    best_aurc: float
    any_confidence_informative: bool        # any model with aurc_gain > eps

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "n_questions": self.n_questions,
            "k": self.k,
            "coverages": list(self.coverages),
            "models": list(self.models),
            "per_model": [p.to_dict() for p in self.per_model],
            "best_aurc_model": self.best_aurc_model,
            "best_aurc": self.best_aurc,
            "any_confidence_informative": self.any_confidence_informative,
        }


def analyze(
    matrix: dict,
    *,
    benchmark: str | None = None,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> SelectiveSummary:
    """Per-model risk-coverage / selective-prediction from an ``oracle_matrix`` dict.

    Confidence is self-consistency ``max(p_hat, 1 - p_hat)``; correctness is the
    majority-vote outcome ``solves >= K//2 + 1``. AURC is the mean selective risk over all
    coverage levels; ``aurc_gain`` compares it to the random-ordering baseline (the base
    error rate), so a positive gain means the confidence signal is genuinely informative.
    """
    solves, k, models = solve_counts(matrix)
    bench = str(benchmark or matrix.get("benchmark", "?"))
    q, m = (solves.shape[0], solves.shape[1]) if solves.ndim == 2 else (0, 0)
    covs = list(coverages)
    if q == 0 or m == 0 or k == 0:
        return SelectiveSummary(bench, 0, k, covs, list(models), [], None, 0.0, False)

    p_hat = solves / k
    confidence = np.maximum(p_hat, 1.0 - p_hat)                 # (Q, M) self-consistency
    majority_correct = (solves >= (k // 2 + 1)).astype(float)   # (Q, M)

    per_model: list[ModelSelective] = []
    for i in range(m):
        correct = majority_correct[:, i]
        _, sel_acc = risk_coverage(correct, confidence[:, i])
        base_acc = float(correct.mean())
        aurc = float((1.0 - sel_acc).mean())
        random_aurc = 1.0 - base_acc
        acc_at = {c: _accuracy_at(sel_acc, c) for c in covs}
        # abstention_gain is the MILDEST abstention the field documents
        # (acc@0.8 - acc@1.0): the accuracy lift from dropping only the least-
        # confident queries. That is the HIGHEST partial coverage below full, so
        # take max(), not min() -- min() reports the DEEPEST level (acc@0.5), a
        # far more aggressive abstention than the field claims. Degenerates
        # correctly for a single partial level and for coverages=(1.0,) (gain 0).
        partial = max((c for c in covs if c < 1.0), default=1.0)
        abstention_gain = acc_at.get(partial, base_acc) - acc_at.get(1.0, base_acc)
        per_model.append(ModelSelective(
            model=models[i], base_accuracy=base_acc, aurc=aurc, random_aurc=random_aurc,
            aurc_gain=random_aurc - aurc, accuracy_at_coverage=acc_at,
            abstention_gain=abstention_gain,
        ))

    best = min(per_model, key=lambda p: p.aurc)
    return SelectiveSummary(
        benchmark=bench, n_questions=q, k=k, coverages=covs, models=list(models),
        per_model=per_model, best_aurc_model=best.model, best_aurc=best.aurc,
        any_confidence_informative=any(p.aurc_gain > _USEFUL_EPS for p in per_model),
    )


def render(summary: SelectiveSummary) -> str:
    """Markdown: per-model risk-coverage table + the self-consistency-usefulness verdict."""
    s = summary
    out = ["# Selective prediction (self-consistency risk-coverage)\n"]
    if s.n_questions == 0:
        return "".join(out) + "\n_(no matrix data)_\n"
    out.append(f"n = {s.n_questions} questions, K = {s.k} samples/model · confidence = "
               "self-consistency max(p̂, 1−p̂)\n")
    cov_cols = " | ".join(f"acc@{c:g}" for c in s.coverages)
    out.append(f"| model | base acc | AURC | AURC gain vs random | {cov_cols} | abstain gain |")
    out.append("|---|---|---|---|" + "---|" * len(s.coverages) + "---|")
    for p in s.per_model:
        cov_vals = " | ".join(f"{p.accuracy_at_coverage[c]:.3f}" for c in s.coverages)
        out.append(f"| {p.model} | {p.base_accuracy:.3f} | {p.aurc:.3f} | {p.aurc_gain:+.3f} "
                   f"| {cov_vals} | {p.abstention_gain:+.3f} |")
    out.append(f"\n- best AURC (most-calibrated selective risk): **{s.best_aurc_model}** "
               f"({s.best_aurc:.3f})")
    if s.any_confidence_informative:
        out.append("\n**Verdict:** self-consistency confidence IS informative (AURC beats the "
                   "random-ordering baseline) — abstaining on low-agreement queries lifts "
                   "accuracy, the offline signal a UCCI confidence cascade would exploit.")
    else:
        out.append("\n**Verdict:** self-consistency confidence does NOT beat random ordering "
                   "here — selective abstention buys no accuracy, so a confidence cascade "
                   "would not help on this matrix.")
    return "\n".join(out) + "\n"
