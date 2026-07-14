"""Oracle-matrix <-> rigorous-eval collection-integrity reconciliation.

``docs/ORACLE_CEILING_DIAGNOSTIC.md`` §5.3 (integrity guard #3) mandates a decisive
collection-correctness check that ``scripts/oracle_ceiling.py`` never performs:

    "Cross-check against the rigorous aggregates: the per-query matrix, averaged per
     model, must reproduce the rigorous eval numbers (glm 0.794, deepseek 0.747, kimi
     0.742 on math; etc.) within CI. If it does not, the matrix collection is buggy and
     the verdict is void. This is a cheap, decisive collection-correctness check."

Every oracle-ceiling / union-oracle / sampling / complementarity conclusion rests on
``oracle_matrix_<bench>.json``. This module cross-checks that matrix against the
independent ``<bench>_rigorous.json`` per-model accuracies and emits a
``TRUSTWORTHY / SUSPECT / VOID`` verdict, so a silently mis-collected matrix cannot
masquerade as a confident answer.

The per-model check is a **combined-uncertainty** z-test. Both artifacts estimate the
same quantity — the model's per-sample accuracy — but each carries its own noise: the
matrix mean has the standard error of a per-query average (which *dominates* at the
small per-benchmark n), and the rigorous mean has its across-reps std
(``single_std::<model>``). Comparing the matrix mean against the tiny rigorous std alone
(ignoring the matrix's own SE) manufactures phantom disagreements; the reconciliation
combines both, ``sigma = hypot(matrix_se, rigorous_std)``.

Distinct from ``verify_benchmark`` (#174, hidden-benchmark BUILD integrity: hashes and
counts), ``dataset_quality`` (#192, a built benchmark's data quality), and
``oracle_ceiling`` itself (computes the ceiling but never validates its own input
matrix). Reuses ``sampling.solve_counts`` so it can't drift from the matrix decoder.
Pure numpy/stdlib over on-disk JSON — no torch, no network, no GPU.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trinity.analysis.sampling import solve_counts

__all__ = [
    "PerModelReconciliation",
    "ReconciliationSummary",
    "matrix_accuracy",
    "rigorous_accuracy",
    "reconcile",
    "reconcile_files",
    "render",
]

# Overall verdicts.
TRUSTWORTHY = "TRUSTWORTHY"
SUSPECT = "SUSPECT"
VOID = "VOID"

# Per-model statuses.
_MATCH = "reconciled"
_SUSPECT = "suspect"
_VOID = "void"

DEFAULT_Z_OK = 2.0     # |z| <= this -> per-model reconciled
DEFAULT_Z_VOID = 3.0   # |z| >  this -> per-model void (else suspect)

_SINGLE = "single::"
_SINGLE_STD = "single_std::"

_ORDER = {TRUSTWORTHY: 0, SUSPECT: 1, VOID: 2}


def _escalate(current: str, target: str) -> str:
    """Return the more severe of two verdicts."""
    return target if _ORDER[target] > _ORDER[current] else current


def matrix_accuracy(
    matrix: dict,
) -> tuple[dict[str, float], dict[str, float], int, int, list[str]]:
    """Per-model ``(accuracy, standard_error, K, n, models)`` from an ``oracle_matrix`` dict.

    ``accuracy[m]`` is the matrix's estimate of the model's per-sample accuracy — the mean
    over the ``n`` queries of ``solves[q, m] / K``. ``se[m]`` is the standard error of that
    mean over the queries (sample std / ``sqrt(n)``), i.e. the matrix estimate's own
    uncertainty, which the reconciliation must account for. For a mean this normal-approx
    SE equals the query-bootstrap SE, so it stays deterministic (no RNG). Reuses
    ``sampling.solve_counts`` so it can't drift from the schema decoder.
    """
    solves, k, models = solve_counts(matrix)
    n = int(solves.shape[0]) if solves.ndim == 2 else 0
    acc: dict[str, float] = {}
    se: dict[str, float] = {}
    if n == 0 or k == 0 or not models:
        return acc, se, k, n, list(models)
    p = solves / k                                        # per-query mean-of-K solve rate
    means = p.mean(axis=0)
    stds = p.std(axis=0, ddof=1) if n > 1 else np.zeros(len(models))
    for i, m in enumerate(models):
        acc[m] = float(means[i])
        se[m] = float(stds[i] / math.sqrt(n))
    return acc, se, k, n, list(models)


def rigorous_accuracy(
    rigorous: dict,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Per-model ``(accuracy, across_reps_std, models)`` from a ``<bench>_rigorous.json`` dict.

    Accepts either the whole file (``{"results": {...}}``) or the ``results`` block itself.
    Parses ``single::<model>`` (per-sample accuracy) and ``single_std::<model>`` (std across
    reps); a model with no reported std defaults to ``0.0`` (treated as exact).
    """
    results = rigorous.get("results", rigorous) if isinstance(rigorous, dict) else {}
    acc: dict[str, float] = {}
    std: dict[str, float] = {}
    for key, val in results.items():
        if not isinstance(key, str):
            continue
        if key.startswith(_SINGLE_STD):                  # check the longer prefix first
            std[key[len(_SINGLE_STD):]] = float(val)
        elif key.startswith(_SINGLE):
            acc[key[len(_SINGLE):]] = float(val)
    return acc, {m: std.get(m, 0.0) for m in acc}, list(acc)


@dataclass(frozen=True)
class PerModelReconciliation:
    """One model's matrix-vs-rigorous agreement (a combined-uncertainty z-test)."""

    model: str
    matrix_acc: float
    matrix_se: float
    rigorous_acc: float
    rigorous_std: float
    combined_sigma: float
    z: float
    status: str            # reconciled / suspect / void

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "model": self.model,
            "matrix_acc": self.matrix_acc,
            "matrix_se": self.matrix_se,
            "rigorous_acc": self.rigorous_acc,
            "rigorous_std": self.rigorous_std,
            "combined_sigma": self.combined_sigma,
            "z": self.z,
            "status": self.status,
        }


@dataclass(frozen=True)
class ReconciliationSummary:
    """Whole-benchmark reconciliation of the collected matrix against the rigorous eval."""

    benchmark: str
    k: int
    n: int
    models: list[str]                       # models shared by both artifacts
    per_model: list[PerModelReconciliation]
    models_match: bool
    matrix_only: list[str]                  # models in the matrix but not the rigorous eval
    rigorous_only: list[str]                # models in the rigorous eval but not the matrix
    best_model_matrix: str | None
    best_model_rigorous: str | None
    best_model_agrees: bool
    rank_agrees: bool
    z_ok: float
    z_void: float
    verdict: str                            # TRUSTWORTHY / SUSPECT / VOID
    problems: list[str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "benchmark": self.benchmark,
            "k": self.k,
            "n": self.n,
            "models": list(self.models),
            "per_model": [p.to_dict() for p in self.per_model],
            "models_match": self.models_match,
            "matrix_only": list(self.matrix_only),
            "rigorous_only": list(self.rigorous_only),
            "best_model_matrix": self.best_model_matrix,
            "best_model_rigorous": self.best_model_rigorous,
            "best_model_agrees": self.best_model_agrees,
            "rank_agrees": self.rank_agrees,
            "z_ok": self.z_ok,
            "z_void": self.z_void,
            "verdict": self.verdict,
            "problems": list(self.problems),
        }


def reconcile(
    matrix: dict,
    rigorous: dict,
    *,
    benchmark: str | None = None,
    z_ok: float = DEFAULT_Z_OK,
    z_void: float = DEFAULT_Z_VOID,
) -> ReconciliationSummary:
    """Reconcile an ``oracle_matrix`` dict against a ``<bench>_rigorous.json`` dict.

    For each shared model, ``z = (matrix_acc - rigorous_acc) / hypot(matrix_se,
    rigorous_std)``; ``|z| <= z_ok`` reconciles, ``z_ok < |z| <= z_void`` is suspect, and
    ``|z| > z_void`` is void. The overall verdict is the worst per-model status, escalated
    to at least SUSPECT on a model-set mismatch or a best-model disagreement, and VOID when
    the two artifacts share no models at all.
    """
    macc, mse, k, n, mmodels = matrix_accuracy(matrix)
    racc, rstd, rmodels = rigorous_accuracy(rigorous)
    bench = str(benchmark or matrix.get("benchmark") or rigorous.get("benchmark") or "?")

    shared = [m for m in mmodels if m in racc]
    matrix_only = [m for m in mmodels if m not in racc]
    rigorous_only = [m for m in rmodels if m not in macc]
    models_match = bool(shared) and not matrix_only and not rigorous_only

    problems: list[str] = []
    per_model: list[PerModelReconciliation] = []
    verdict = TRUSTWORTHY
    for m in shared:
        sigma = math.hypot(mse[m], rstd[m])
        diff = macc[m] - racc[m]
        if sigma > 0:
            z = diff / sigma
        else:
            z = 0.0 if abs(diff) < 1e-12 else math.copysign(math.inf, diff)
        az = abs(z)
        if az > z_void:
            status = _VOID
            verdict = _escalate(verdict, VOID)
        elif az > z_ok:
            status = _SUSPECT
            verdict = _escalate(verdict, SUSPECT)
        else:
            status = _MATCH
        per_model.append(
            PerModelReconciliation(m, macc[m], mse[m], racc[m], rstd[m], sigma, z, status)
        )
        if status != _MATCH:
            problems.append(f"{m}: matrix {macc[m]:.4f} vs rigorous {racc[m]:.4f} "
                            f"(z={z:+.2f}, {status})")

    best_model_matrix = max(shared, key=lambda m: macc[m]) if shared else None
    best_model_rigorous = max(shared, key=lambda m: racc[m]) if shared else None
    best_model_agrees = best_model_matrix is not None and best_model_matrix == best_model_rigorous
    rank_agrees = (
        sorted(shared, key=lambda m: macc[m]) == sorted(shared, key=lambda m: racc[m])
        if shared else True
    )

    if matrix_only:
        problems.append(f"models in matrix but not rigorous: {matrix_only}")
    if rigorous_only:
        problems.append(f"models in rigorous but not matrix: {rigorous_only}")
    if not shared and (mmodels or rmodels):
        # data present on at least one side but the model sets are disjoint -> void.
        # Both sides empty is a benign no-data case (verdict stays TRUSTWORTHY).
        problems.append("no shared models between matrix and rigorous — cannot reconcile")
        verdict = VOID
    if matrix_only or rigorous_only:
        verdict = _escalate(verdict, SUSPECT)
    if shared and not best_model_agrees:
        problems.append(f"best model disagrees: matrix={best_model_matrix} "
                        f"rigorous={best_model_rigorous}")
        verdict = _escalate(verdict, SUSPECT)

    return ReconciliationSummary(
        benchmark=bench, k=k, n=n, models=list(shared), per_model=per_model,
        models_match=models_match, matrix_only=matrix_only, rigorous_only=rigorous_only,
        best_model_matrix=best_model_matrix, best_model_rigorous=best_model_rigorous,
        best_model_agrees=best_model_agrees, rank_agrees=rank_agrees,
        z_ok=z_ok, z_void=z_void, verdict=verdict, problems=problems,
    )


def reconcile_files(
    matrix_path: str | Path,
    rigorous_path: str | Path,
    **kwargs: Any,
) -> ReconciliationSummary:
    """Load a matrix + rigorous JSON pair from disk and reconcile them."""
    matrix = json.loads(Path(matrix_path).read_text())
    rigorous = json.loads(Path(rigorous_path).read_text())
    return reconcile(matrix, rigorous, **kwargs)


_VERDICT_LINE = {
    TRUSTWORTHY: "**Verdict: TRUSTWORTHY** — the matrix reproduces the rigorous aggregates "
                 "within CI; downstream oracle-ceiling conclusions rest on a validated matrix.",
    SUSPECT: "**Verdict: SUSPECT** — a model drifts beyond the tolerance band or the model "
             "sets disagree; inspect the matrix collection before trusting the ceiling.",
    VOID: "**Verdict: VOID** — the matrix does not reproduce the rigorous eval; per §5.3 the "
          "collection is buggy and any oracle-ceiling verdict built on it is void.",
}


def render(summary: ReconciliationSummary) -> str:
    """Markdown: per-model reconciliation table + the TRUSTWORTHY/SUSPECT/VOID verdict."""
    s = summary
    out = [f"# Matrix-vs-rigorous reconciliation — {s.benchmark}\n"]
    if not s.per_model and not s.problems:
        return "".join(out) + "\n_(no data to reconcile)_\n"
    out.append(f"n = {s.n} queries, K = {s.k} samples/model · tolerance |z| <= {s.z_ok:g} "
               f"(suspect > {s.z_ok:g}, void > {s.z_void:g})\n")
    if s.per_model:
        out.append("| model | matrix acc | rigorous acc | combined σ | z | status |")
        out.append("|---|---|---|---|---|---|")
        for p in s.per_model:
            out.append(f"| {p.model} | {p.matrix_acc:.4f} ± {p.matrix_se:.4f} "
                       f"| {p.rigorous_acc:.4f} ± {p.rigorous_std:.4f} | {p.combined_sigma:.4f} "
                       f"| {p.z:+.2f} | {p.status} |")
        out.append(f"\n- best model: matrix **{s.best_model_matrix}** vs rigorous "
                   f"**{s.best_model_rigorous}** — {'agree' if s.best_model_agrees else 'DISAGREE'}")
        out.append(f"- model ranking: {'agrees' if s.rank_agrees else 'DISAGREES'}")
    out.append("\n" + _VERDICT_LINE[s.verdict])
    if s.problems:
        out.append("\nProblems:")
        out.extend(f"  - {p}" for p in s.problems)
    return "\n".join(out) + "\n"
