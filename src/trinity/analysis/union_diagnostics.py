"""Cross-benchmark (equal-weight union) view of the per-model sampling & selective diagnostics.

Why this exists
---------------
The competition score is the **equal-weighted union of the 3 benchmarks** —
``pr_eval`` averages the per-benchmark scores (``sum(...) / len(per_benchmark)``), and
:mod:`trinity.analysis.union_oracle` already aggregates the oracle ceiling that way because
"is the routing win cross-task?" cannot be answered one benchmark at a time.

The other two per-model diagnostics never got that view. :mod:`trinity.analysis.sampling`
(pass@1 / pass@K / majority@K) and :mod:`trinity.analysis.selective` (risk-coverage / AURC)
are strictly per-matrix, so their headline questions are only ever answered per benchmark:

* **sampling** — *does cheaply re-sampling the best single model rival the routing oracle?*
  A model can rival the ceiling on math500 and collapse on mmlu; per-benchmark answers can
  disagree, and it is the union answer that decides whether the router is worth tuning.
* **selective** — *is self-consistency confidence informative enough to abstain on?* Same
  problem: calibration that only holds on one task is not a usable abstention policy.

This module answers both across the union. It never re-derives the per-model math — it calls
the canonical ``sampling.analyze`` / ``selective.analyze`` per matrix and equal-weight
averages their outputs, and it reuses their own verdict thresholds — so the union view
cannot drift from the per-benchmark reports it summarises.

Model sets must match across benchmarks (the same rule :func:`union_oracle.union_oracle`
enforces): an equal-weight per-model average over a model that is missing from one benchmark
would be taken over its own, often favourable, subset of tasks and is not comparable.

Pure / offline — numpy + stdlib over ``oracle_matrix_<bench>.json`` files already on disk.
No torch, no network, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from trinity.analysis import selective as _selective
from trinity.analysis.sampling import analyze as analyze_sampling
from trinity.analysis.selective import DEFAULT_COVERAGES
from trinity.analysis.selective import analyze as analyze_selective

__all__ = [
    "UnionModelSampling",
    "UnionSamplingSummary",
    "UnionModelSelective",
    "UnionSelectiveSummary",
    "union_sampling",
    "union_selective",
    "render_sampling",
    "render_selective",
]

#: Mirrors ``sampling.analyze``'s own rivalry test, so the union verdict uses the same rule.
_RIVAL_EPS = 1e-9


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _require_shared_models(summaries: Sequence[Any]) -> list[str]:
    """The model list common to every summary, or raise (mirrors ``union_oracle``).

    Raises:
        ValueError: if any benchmark carries a different model set — an equal-weight
            per-model average would then be over a different subset per model.
    """
    covered = [s for s in summaries if s.n_questions > 0]
    if not covered:
        return []
    models = list(covered[0].models)
    for s in covered[1:]:
        if list(s.models) != models:
            raise ValueError(
                f"benchmark {s.benchmark!r} has models {list(s.models)}, expected {models}"
            )
    return models


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnionModelSampling:
    """One model's sampling metrics, equal-weight averaged across benchmarks."""

    model: str
    pass_at_1: float
    pass_at_k: float
    majority_at_k: float
    self_consistency_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "majority_at_k": self.majority_at_k,
            "self_consistency_gain": self.self_consistency_gain,
        }


@dataclass(frozen=True)
class UnionSamplingSummary:
    """Equal-weight union of the per-benchmark sampling summaries."""

    benchmarks: list[str]
    n_benchmarks: int
    models: list[str]
    per_model: list[UnionModelSampling]
    best_pass1_model: Optional[str]
    best_pass1: float
    best_majority_model: Optional[str]
    best_majority: float
    routing_oracle: float
    majority_rivals_oracle: bool
    per_benchmark_rivals: dict[str, bool]

    @property
    def verdict_is_unanimous(self) -> bool:
        """True when every benchmark agrees with the union rivalry verdict.

        A split here is the whole reason this view exists: a per-benchmark report that
        says "majority rivals the oracle" on one task and not another cannot be read as a
        statement about the composite.
        """
        return all(v == self.majority_rivals_oracle for v in self.per_benchmark_rivals.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmarks": list(self.benchmarks),
            "n_benchmarks": self.n_benchmarks,
            "models": list(self.models),
            "per_model": [p.to_dict() for p in self.per_model],
            "best_pass1_model": self.best_pass1_model,
            "best_pass1": self.best_pass1,
            "best_majority_model": self.best_majority_model,
            "best_majority": self.best_majority,
            "routing_oracle": self.routing_oracle,
            "majority_rivals_oracle": self.majority_rivals_oracle,
            "per_benchmark_rivals": dict(self.per_benchmark_rivals),
            "verdict_is_unanimous": self.verdict_is_unanimous,
        }


def union_sampling(matrices: Sequence[dict]) -> UnionSamplingSummary:
    """Equal-weight union of ``sampling.analyze`` over several ``oracle_matrix`` dicts.

    Args:
        matrices: One ``oracle_matrix_<bench>.json`` dict per benchmark.

    Returns:
        The :class:`UnionSamplingSummary`. Benchmarks with no questions are listed but
        contribute nothing to the averages.

    Raises:
        ValueError: if the covered benchmarks do not share one model set.
    """
    summaries = [analyze_sampling(m) for m in matrices]
    models = _require_shared_models(summaries)
    covered = [s for s in summaries if s.n_questions > 0]
    names = [s.benchmark for s in summaries]

    if not covered or not models:
        return UnionSamplingSummary(
            benchmarks=names, n_benchmarks=0, models=list(models), per_model=[],
            best_pass1_model=None, best_pass1=0.0, best_majority_model=None,
            best_majority=0.0, routing_oracle=0.0, majority_rivals_oracle=False,
            per_benchmark_rivals={},
        )

    by_model = {m: [next(p for p in s.per_model if p.model == m) for s in covered]
                for m in models}
    per_model = [
        UnionModelSampling(
            model=m,
            pass_at_1=_mean([p.pass_at_1 for p in ps]),
            pass_at_k=_mean([p.pass_at_k for p in ps]),
            majority_at_k=_mean([p.majority_at_k for p in ps]),
            self_consistency_gain=_mean([p.self_consistency_gain for p in ps]),
        )
        for m, ps in by_model.items()
    ]

    best_p1 = max(per_model, key=lambda p: p.pass_at_1)
    best_maj = max(per_model, key=lambda p: p.majority_at_k)
    routing_oracle = _mean([s.routing_oracle for s in covered])
    return UnionSamplingSummary(
        benchmarks=names,
        n_benchmarks=len(covered),
        models=list(models),
        per_model=per_model,
        best_pass1_model=best_p1.model,
        best_pass1=best_p1.pass_at_1,
        best_majority_model=best_maj.model,
        best_majority=best_maj.majority_at_k,
        routing_oracle=routing_oracle,
        majority_rivals_oracle=bool(best_maj.majority_at_k >= routing_oracle - _RIVAL_EPS),
        per_benchmark_rivals={s.benchmark: s.majority_rivals_oracle for s in covered},
    )


def render_sampling(summary: UnionSamplingSummary) -> str:
    """Markdown: per-model union table + the cross-task rivalry verdict."""
    out = ["# Cross-benchmark sampling (equal-weight union)\n"]
    if summary.n_benchmarks == 0:
        return "".join(out) + "\n_(no benchmark matrices)_\n"

    out.append(f"Union over {summary.n_benchmarks} benchmark(s): {summary.benchmarks}\n")
    out.append("| model | pass@1 | pass@K | majority@K | self-consistency gain |")
    out.append("|---|---|---|---|---|")
    for p in sorted(summary.per_model, key=lambda x: -x.majority_at_k):
        out.append(f"| {p.model} | {p.pass_at_1:.3f} | {p.pass_at_k:.3f} | "
                   f"{p.majority_at_k:.3f} | {p.self_consistency_gain:+.3f} |")

    out.append(f"\n- **best majority@K**: {summary.best_majority:.3f} "
               f"({summary.best_majority_model})")
    out.append(f"- **union routing oracle**: {summary.routing_oracle:.3f}")
    verdict = ("re-sampling the best single model RIVALS the routing ceiling across the "
               "union — the router has little left to win"
               if summary.majority_rivals_oracle else
               "the routing ceiling still beats re-sampling the best single model across "
               "the union — routing has real headroom")
    out.append(f"\n**Verdict:** {verdict}.")
    if not summary.verdict_is_unanimous:
        split = ", ".join(f"{b}: {'rivals' if v else 'does not rival'}"
                          for b, v in sorted(summary.per_benchmark_rivals.items()))
        out.append(f"\n_Per-benchmark verdicts DISAGREE ({split}) — the union answer is the "
                   "one that matches the composite score._")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# selective
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnionModelSelective:
    """One model's selective-prediction metrics, equal-weight averaged across benchmarks."""

    model: str
    base_accuracy: float
    aurc: float
    random_aurc: float
    aurc_gain: float
    abstention_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_accuracy": self.base_accuracy,
            "aurc": self.aurc,
            "random_aurc": self.random_aurc,
            "aurc_gain": self.aurc_gain,
            "abstention_gain": self.abstention_gain,
        }


@dataclass(frozen=True)
class UnionSelectiveSummary:
    """Equal-weight union of the per-benchmark selective summaries."""

    benchmarks: list[str]
    n_benchmarks: int
    models: list[str]
    per_model: list[UnionModelSelective]
    best_aurc_model: Optional[str]
    best_aurc: float
    any_confidence_informative: bool
    per_benchmark_informative: dict[str, bool]

    @property
    def informative_everywhere(self) -> bool:
        """True when confidence is informative on EVERY benchmark, not just on average.

        An abstention policy that only calibrates on one task is not usable, so the union
        mean alone would overstate it.
        """
        return bool(self.per_benchmark_informative) and all(
            self.per_benchmark_informative.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmarks": list(self.benchmarks),
            "n_benchmarks": self.n_benchmarks,
            "models": list(self.models),
            "per_model": [p.to_dict() for p in self.per_model],
            "best_aurc_model": self.best_aurc_model,
            "best_aurc": self.best_aurc,
            "any_confidence_informative": self.any_confidence_informative,
            "per_benchmark_informative": dict(self.per_benchmark_informative),
            "informative_everywhere": self.informative_everywhere,
        }


def union_selective(
    matrices: Sequence[dict],
    *,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> UnionSelectiveSummary:
    """Equal-weight union of ``selective.analyze`` over several ``oracle_matrix`` dicts.

    Raises:
        ValueError: if the covered benchmarks do not share one model set.
    """
    summaries = [analyze_selective(m, coverages=coverages) for m in matrices]
    models = _require_shared_models(summaries)
    covered = [s for s in summaries if s.n_questions > 0]
    names = [s.benchmark for s in summaries]

    if not covered or not models:
        return UnionSelectiveSummary(
            benchmarks=names, n_benchmarks=0, models=list(models), per_model=[],
            best_aurc_model=None, best_aurc=0.0, any_confidence_informative=False,
            per_benchmark_informative={},
        )

    by_model = {m: [next(p for p in s.per_model if p.model == m) for s in covered]
                for m in models}
    per_model = [
        UnionModelSelective(
            model=m,
            base_accuracy=_mean([p.base_accuracy for p in ps]),
            aurc=_mean([p.aurc for p in ps]),
            random_aurc=_mean([p.random_aurc for p in ps]),
            aurc_gain=_mean([p.aurc_gain for p in ps]),
            abstention_gain=_mean([p.abstention_gain for p in ps]),
        )
        for m, ps in by_model.items()
    ]

    best = min(per_model, key=lambda p: p.aurc)          # lower AURC is better
    return UnionSelectiveSummary(
        benchmarks=names,
        n_benchmarks=len(covered),
        models=list(models),
        per_model=per_model,
        best_aurc_model=best.model,
        best_aurc=best.aurc,
        # Same threshold ``selective.analyze`` uses, referenced rather than copied.
        any_confidence_informative=any(p.aurc_gain > _selective._USEFUL_EPS for p in per_model),
        per_benchmark_informative={s.benchmark: s.any_confidence_informative for s in covered},
    )


def render_selective(summary: UnionSelectiveSummary) -> str:
    """Markdown: per-model union table + whether confidence is usable cross-task."""
    out = ["# Cross-benchmark selective prediction (equal-weight union)\n"]
    if summary.n_benchmarks == 0:
        return "".join(out) + "\n_(no benchmark matrices)_\n"

    out.append(f"Union over {summary.n_benchmarks} benchmark(s): {summary.benchmarks}\n")
    out.append("| model | base acc | AURC | random AURC | AURC gain | abstention gain |")
    out.append("|---|---|---|---|---|---|")
    for p in sorted(summary.per_model, key=lambda x: x.aurc):
        out.append(f"| {p.model} | {p.base_accuracy:.3f} | {p.aurc:.3f} | "
                   f"{p.random_aurc:.3f} | {p.aurc_gain:+.3f} | {p.abstention_gain:+.3f} |")

    out.append(f"\n- **best (lowest) AURC**: {summary.best_aurc:.3f} ({summary.best_aurc_model})")
    verdict = ("self-consistency confidence is informative on the union — abstention is "
               "worth using" if summary.any_confidence_informative else
               "self-consistency confidence is NOT informative on the union — abstention "
               "would not pay")
    out.append(f"\n**Verdict:** {verdict}.")
    if summary.any_confidence_informative and not summary.informative_everywhere:
        weak = ", ".join(b for b, v in sorted(summary.per_benchmark_informative.items()) if not v)
        out.append(f"\n_But confidence is uninformative on: {weak} — an abstention policy "
                   "tuned on the union would misfire there._")
    return "\n".join(out) + "\n"
