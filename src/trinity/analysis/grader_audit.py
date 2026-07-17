"""Grader audit — estimate the fixed grader's error rate on a benchmark's references.

Why this exists
---------------
``docs/ORACLE_CEILING_DIAGNOSTIC.md`` §5 lists five integrity guards that keep a bug in
the diagnostic from producing a confident wrong answer. Four are implemented (reuse the
fixed grader, rigorous cross-check → :mod:`trinity.analysis.reconcile`, threshold
sensitivity → ``scripts/oracle_ceiling.py``, held-out only). **Guard #2 — "review a random
sample of ~30 grading decisions per benchmark, especially boundary cases, and report an
estimated grader error rate. Grader false-negatives deflate the oracle; false-positives
inflate it." — has no implementation.** The whole oracle verdict rests on grader
trustworthiness, and the ``docs/JOURNAL.md`` record is a long list of *format* grading bugs
(``\\mathbf{5}`` vs ``5``, ``90^{\\circ}`` vs ``90``, ``**B**`` vs ``B``, a boxed choice the
extractor missed) that each silently scored a correct answer 0.

This module bounds that confound **offline**, with no torch / model / network. It never
re-implements grading — it drives the FIXED grader (``reward.score_text``) and the read-only
:func:`trinity.grading_explain.explain_grade` tracer — so it cannot itself reintroduce the
brittle-extraction bug guard #1 warns about. For each reference it runs three probes whose
correctness is a property of the grader, not of a second extractor:

* **Self-consistency** — the canonical gold form (``\\boxed{ref}`` for math, the bare letter
  for choice) MUST grade ``1.0`` against its own reference. A failure is a *definite* grader
  false-negative (it cannot recognise its own gold), the strongest signal there is.
* **Fragility** — a set of *semantics-preserving* perturbations of that canonical form
  (surrounding prose, ``$`` wrap, whitespace, Markdown/LaTeX emphasis) MUST keep the grade
  at ``1.0``. Each perturbation is one the grader's own normaliser documents as
  value-preserving, so a flip to ``0.0`` is unambiguous grader fragility — a false-negative
  waiting for a model to phrase its answer that way.
* **False-positive** — a clearly *different* value (``ref + 1`` for a numeric answer, a
  different option letter) MUST grade ``0.0``. Acceptance is a grader false-positive that
  would inflate the oracle.

The headline per-benchmark ``estimated_error_rate`` is the fraction of auditable references
that tripped ANY probe; ``false_negative_rate`` (self-consistency + fragility) and
``false_positive_rate`` are reported separately, matching the guard's deflate/inflate
framing. Boundary cases — the references that tripped a probe — are surfaced with their
``explain_grade`` trace for the mandated human review.

Code benchmarks (LiveCodeBench / BigCodeBench) grade by executing tests, not by extracting a
string, so this extraction-format audit does not cover them; they are counted and skipped.

Pure / deterministic (seeded sampling) / offline — numpy-free, stdlib + the grader only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from trinity.grading_explain import explain_grade
from trinity.orchestration.reward import (
    CHOICE_BENCHMARKS,
    CODE_BENCHMARKS,
    MATH_BENCHMARKS,
    extract_boxed,
    normalize_reference_letter,
    resolve_benchmark,
    score_text,
)

__all__ = [
    "ProbeFinding",
    "ItemAudit",
    "BenchmarkAudit",
    "audit_item",
    "audit",
    "render",
]

#: Option letters used to synthesise a "clearly wrong" choice for the false-positive probe.
#: A different letter than the gold is wrong for a single-correct-answer item regardless of
#: how many options the question actually had.
_CHOICE_POOL = "ABCD"


def benchmark_kind(benchmark: str) -> str:
    """Map a benchmark name to ``"math"`` / ``"choice"`` / ``"code"`` / ``"unknown"``.

    Uses the grader's OWN benchmark sets (via :func:`reward.resolve_benchmark`) so the audit
    can never disagree with which grading path ``score_text`` will take.
    """
    key = resolve_benchmark(benchmark)
    if key in MATH_BENCHMARKS:
        return "math"
    if key in CHOICE_BENCHMARKS:
        return "choice"
    if key in CODE_BENCHMARKS:
        return "code"
    return "unknown"


def _is_plain_number(s: str) -> Optional[float]:
    """Return the float value of ``s`` if it is a bare int/decimal (no LaTeX), else ``None``.

    Kept deliberately strict: a value is only used to synthesise the numeric wrong-answer
    probe when it parses cleanly, so the "clearly wrong" variant (``value + 1``) is provably
    different. Fractions / expressions / set answers return ``None`` and simply skip the
    false-positive probe rather than risk an invalid one.
    """
    t = s.strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _math_gold_core(reference: object) -> Optional[str]:
    """The bare gold answer string for a math reference (unboxed), or ``None`` if empty."""
    ref = reference if isinstance(reference, str) else ("" if reference is None else str(reference))
    core = extract_boxed(ref)
    core = core if core is not None else ref
    core = core.strip()
    return core or None


def _math_forms(core: str) -> tuple[list[str], list[str]]:
    """(semantics-preserving perturbations, clearly-wrong variants) for a math gold ``core``.

    The perturbations are exactly the value-preserving transforms ``normalize_math_answer``
    documents (prose lead-in, ``$`` wrap, whitespace, a ``\\text{}`` font wrap) — a correct
    grader is invariant to every one. The ``\\text{}`` wrap is only added when ``core`` has no
    inner braces, since the font regex only unwraps a brace-free payload.
    """
    boxed = f"\\boxed{{{core}}}"
    perturbations = [
        f"The final answer is {boxed}.",
        f"${boxed}$",
        f"\n\n{boxed}\n",
        f"So the answer is {boxed}",
    ]
    if "{" not in core and "}" not in core:
        perturbations.append(f"\\boxed{{\\text{{{core}}}}}")
    wrong: list[str] = []
    val = _is_plain_number(core)
    if val is not None:
        other = int(val) + 1 if val == int(val) else val + 1.0
        other_s = str(int(other)) if isinstance(other, int) or other == int(other) else str(other)
        wrong.append(f"\\boxed{{{other_s}}}")
    return perturbations, wrong


def _choice_forms(letter: str) -> tuple[list[str], list[str]]:
    """(semantics-preserving perturbations, clearly-wrong variants) for a choice ``letter``.

    Perturbations are the commitment phrasings / emphasis wrappers the JOURNAL bugs were
    about (``**B**``, ``\\boxed{B}``, ``\\textbf{B}``, ``Answer: B``); each must still read as
    ``letter``. The wrong variant is any other pool letter — wrong for a single-correct item.
    """
    perturbations = [
        f"The answer is {letter}.",
        f"({letter})",
        f"**{letter}**",
        f"\\boxed{{{letter}}}",
        f"\\textbf{{{letter}}}",
        f"Answer: {letter}",
    ]
    other = next((c for c in _CHOICE_POOL if c != letter), None)
    wrong = [other] if other is not None else []
    return perturbations, wrong


@dataclass(frozen=True)
class ProbeFinding:
    """One anomaly: a candidate that graded against expectation, with its grade trace."""

    kind: str                       # "self_consistency" | "fragility" | "false_positive"
    candidate: str
    expected: float
    got: float
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "candidate": self.candidate,
            "expected": self.expected,
            "got": self.got,
            "trace": list(self.trace),
        }


@dataclass(frozen=True)
class ItemAudit:
    """Audit of a single reference: the three probes plus any anomalies found."""

    benchmark: str
    kind: str
    reference: str
    auditable: bool                 # False for code/unknown/unresolvable references
    self_consistent: bool           # canonical gold form graded 1.0
    fragility_total: int
    fragility_flips: int            # semantics-preserving perturbations that dropped to 0
    false_positive_total: int
    false_positive_hits: int        # clearly-wrong variants that graded 1
    findings: list[ProbeFinding] = field(default_factory=list)

    @property
    def is_boundary(self) -> bool:
        """Any anomaly at all — the cases the guard says to surface for human review."""
        return self.auditable and (
            not self.self_consistent or self.fragility_flips > 0 or self.false_positive_hits > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "kind": self.kind,
            "reference": self.reference,
            "auditable": self.auditable,
            "self_consistent": self.self_consistent,
            "fragility_total": self.fragility_total,
            "fragility_flips": self.fragility_flips,
            "false_positive_total": self.false_positive_total,
            "false_positive_hits": self.false_positive_hits,
            "is_boundary": self.is_boundary,
            "findings": [f.to_dict() for f in self.findings],
        }


def _traced(benchmark: str, candidate: str, reference: object) -> list[str]:
    """The ``explain_grade`` step trace for one grading decision (best-effort, read-only)."""
    try:
        return explain_grade(benchmark, candidate, reference).steps
    except Exception:               # pragma: no cover - tracer is a courtesy, never load-bearing
        return []


def audit_item(benchmark: str, reference: object) -> ItemAudit:
    """Run the three grader probes for one ``(benchmark, reference)``.

    Non-math/choice benchmarks and references that do not resolve to a gold form are returned
    as ``auditable=False`` (counted, never scored against). Every grade goes through the
    fixed ``score_text``; ``findings`` carries the ``explain_grade`` trace only for anomalies.
    """
    kind = benchmark_kind(benchmark)
    ref_str = "" if reference is None else str(reference)

    if kind == "math":
        core = _math_gold_core(reference)
        canonical = f"\\boxed{{{core}}}" if core is not None else None
        forms = _math_forms(core) if core is not None else ([], [])
    elif kind == "choice":
        letter = normalize_reference_letter(reference)
        canonical = letter
        forms = _choice_forms(letter) if letter is not None else ([], [])
    else:
        canonical = None
        forms = ([], [])

    if canonical is None:
        return ItemAudit(
            benchmark=benchmark, kind=kind, reference=ref_str, auditable=False,
            self_consistent=True, fragility_total=0, fragility_flips=0,
            false_positive_total=0, false_positive_hits=0,
        )

    findings: list[ProbeFinding] = []

    # Probe 1 — self-consistency: the canonical gold form must grade 1.0 against its own ref.
    base = score_text(benchmark, canonical, reference)
    self_consistent = base >= 1.0
    if not self_consistent:
        findings.append(ProbeFinding("self_consistency", canonical, 1.0, base,
                                     _traced(benchmark, canonical, reference)))

    perturbations, wrong = forms

    # Probe 2 — fragility: only meaningful once the baseline itself grades correct.
    frag_total = len(perturbations) if self_consistent else 0
    frag_flips = 0
    if self_consistent:
        for cand in perturbations:
            if score_text(benchmark, cand, reference) < 1.0:
                frag_flips += 1
                findings.append(ProbeFinding("fragility", cand, 1.0, 0.0,
                                             _traced(benchmark, cand, reference)))

    # Probe 3 — false-positive: a clearly-wrong variant must grade 0.0.
    fp_total = len(wrong)
    fp_hits = 0
    for cand in wrong:
        got = score_text(benchmark, cand, reference)
        if got >= 1.0:
            fp_hits += 1
            findings.append(ProbeFinding("false_positive", cand, 0.0, got,
                                         _traced(benchmark, cand, reference)))

    return ItemAudit(
        benchmark=benchmark, kind=kind, reference=ref_str, auditable=True,
        self_consistent=self_consistent, fragility_total=frag_total, fragility_flips=frag_flips,
        false_positive_total=fp_total, false_positive_hits=fp_hits, findings=findings,
    )


@dataclass(frozen=True)
class BenchmarkAudit:
    """Aggregate grader-error estimate for one benchmark, with boundary samples."""

    benchmark: str
    kind: str
    n_seen: int                     # references passed in
    n_auditable: int                # math/choice with a resolvable gold
    n_skipped: int                  # code/unknown/unresolvable
    self_consistency_failures: int
    fragility_total: int
    fragility_flips: int
    false_positive_total: int
    false_positive_hits: int
    n_boundary: int
    boundary_samples: list[ItemAudit] = field(default_factory=list)

    @property
    def false_negative_rate(self) -> Optional[float]:
        """Grader false-negatives: (self-consistency failures + fragility flips) / probes.

        ``None`` when nothing auditable was seen. A false-negative is a correct answer scored
        0 — it *deflates* the oracle.
        """
        denom = self.n_auditable + self.fragility_total
        hits = self.self_consistency_failures + self.fragility_flips
        return hits / denom if denom > 0 else None

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Grader false-positives (a wrong answer scored 1) — *inflates* the oracle."""
        return self.false_positive_hits / self.false_positive_total if self.false_positive_total else None

    @property
    def estimated_error_rate(self) -> Optional[float]:
        """Headline: fraction of auditable references that tripped ANY probe."""
        return self.n_boundary / self.n_auditable if self.n_auditable else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "kind": self.kind,
            "n_seen": self.n_seen,
            "n_auditable": self.n_auditable,
            "n_skipped": self.n_skipped,
            "self_consistency_failures": self.self_consistency_failures,
            "fragility_total": self.fragility_total,
            "fragility_flips": self.fragility_flips,
            "false_positive_total": self.false_positive_total,
            "false_positive_hits": self.false_positive_hits,
            "n_boundary": self.n_boundary,
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "estimated_error_rate": self.estimated_error_rate,
            "boundary_samples": [b.to_dict() for b in self.boundary_samples],
        }


def audit(
    items: Iterable[tuple[str, object]],
    *,
    sample_size: int = 30,
    seed: int = 0,
) -> list[BenchmarkAudit]:
    """Audit ``(benchmark, reference)`` items, grouped per benchmark.

    Args:
        items: iterable of ``(benchmark, reference)`` — e.g. the ``benchmark``/``reference``
            fields of the built hidden-benchmark items.
        sample_size: max boundary cases retained per benchmark for human review (the guard's
            "~30 grading decisions"). A seeded shuffle picks them when there are more.
        seed: RNG seed for the boundary-sample selection (deterministic).

    Returns:
        One :class:`BenchmarkAudit` per benchmark, ordered by first appearance.
    """
    by_bench: dict[str, list[ItemAudit]] = {}
    order: list[str] = []
    for benchmark, reference in items:
        if benchmark not in by_bench:
            by_bench[benchmark] = []
            order.append(benchmark)
        by_bench[benchmark].append(audit_item(benchmark, reference))

    out: list[BenchmarkAudit] = []
    for benchmark in order:
        audits = by_bench[benchmark]
        auditable = [a for a in audits if a.auditable]
        boundary = [a for a in auditable if a.is_boundary]
        rng = random.Random(seed)
        sampled = list(boundary)
        rng.shuffle(sampled)
        sampled = sampled[:sample_size]
        out.append(BenchmarkAudit(
            benchmark=benchmark,
            kind=audits[0].kind if audits else "unknown",
            n_seen=len(audits),
            n_auditable=len(auditable),
            n_skipped=len(audits) - len(auditable),
            self_consistency_failures=sum(1 for a in auditable if not a.self_consistent),
            fragility_total=sum(a.fragility_total for a in auditable),
            fragility_flips=sum(a.fragility_flips for a in auditable),
            false_positive_total=sum(a.false_positive_total for a in auditable),
            false_positive_hits=sum(a.false_positive_hits for a in auditable),
            n_boundary=len(boundary),
            boundary_samples=sampled,
        ))
    return out


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render(audits: list[BenchmarkAudit], *, max_samples: int = 5) -> str:
    """Markdown: the estimated grader-error table + a few boundary traces per benchmark."""
    out = ["# Grader audit (ORACLE_CEILING_DIAGNOSTIC §5 guard #2)\n"]
    if not audits:
        return "".join(out) + "\n_(no references audited)_\n"

    out.append("| benchmark | kind | auditable | est. error | false-neg | false-pos | skipped |")
    out.append("|---|---|---|---|---|---|---|")
    worst = 0.0
    for a in audits:
        out.append(
            f"| {a.benchmark} | {a.kind} | {a.n_auditable} | {_pct(a.estimated_error_rate)} | "
            f"{_pct(a.false_negative_rate)} | {_pct(a.false_positive_rate)} | {a.n_skipped} |"
        )
        worst = max(worst, a.estimated_error_rate or 0.0)

    out.append("")
    for a in audits:
        if not a.boundary_samples:
            continue
        out.append(f"\n## {a.benchmark}: {a.n_boundary} boundary case(s) for review")
        for item in a.boundary_samples[:max_samples]:
            for f in item.findings:
                out.append(
                    f"- **{f.kind}** on ref `{item.reference}` → candidate `{f.candidate}` "
                    f"graded {f.got:.0f} (expected {f.expected:.0f})"
                )

    verdict = (
        "grader looks trustworthy on these references — the oracle is not obviously grader-driven"
        if worst <= 0.02 else
        f"grader tripped on up to {worst:.1%} of a benchmark's references — audit the boundary "
        "cases above before trusting the oracle verdict"
    )
    out.append(f"\n**Verdict:** {verdict}.")
    return "\n".join(out) + "\n"
