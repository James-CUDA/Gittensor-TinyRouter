"""Offline tests for the grader-audit diagnostic (ORACLE_CEILING_DIAGNOSTIC §5 guard #2).

The audit drives the FIXED grader (``reward.score_text``) over three probes per reference —
self-consistency, semantics-preserving fragility, and a clearly-wrong false-positive — and
reports an estimated grader error rate. These tests confirm (a) a clean reference on the real
grader shows zero error with non-trivial probe denominators, and (b) the audit actually
DETECTS fragility / false-positives / self-inconsistency when the grader misbehaves (a fake
grader stands in), so the diagnostic can never silently pass a broken grader. No torch / no
network.
"""
import json
import subprocess
import sys
from pathlib import Path

from trinity.analysis import grader_audit as grader_audit_pkg  # re-export check
from trinity.analysis.grader_audit import (
    audit,
    audit_item,
    benchmark_kind,
    render,
)

_REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# kind resolution + skip rules
# --------------------------------------------------------------------------- #
def test_benchmark_kind_uses_grader_sets():
    assert benchmark_kind("math500") == "math"
    assert benchmark_kind("mmlu") == "choice"
    assert benchmark_kind("livecodebench") == "code"
    assert benchmark_kind("totally-made-up") == "unknown"


def test_code_and_unknown_are_skipped_not_scored():
    for bench in ("livecodebench", "bigcodebench", "totally-made-up"):
        a = audit_item(bench, "anything")
        assert a.auditable is False
        assert a.self_consistent is True          # vacuously — nothing was probed
        assert a.fragility_total == 0 and a.false_positive_total == 0
        assert a.is_boundary is False


def test_empty_reference_is_unauditable():
    assert audit_item("math500", "").auditable is False
    assert audit_item("mmlu", None).auditable is False


# --------------------------------------------------------------------------- #
# clean references on the REAL grader → zero error, real denominators
# --------------------------------------------------------------------------- #
def test_clean_math_reference_shows_no_error():
    a = audit_item("math500", "5")
    assert a.auditable and a.kind == "math" and a.self_consistent
    assert a.fragility_total >= 4 and a.fragility_flips == 0      # probes actually ran
    assert a.false_positive_total == 1 and a.false_positive_hits == 0
    assert a.is_boundary is False


def test_clean_choice_reference_shows_no_error():
    a = audit_item("mmlu", "B")
    assert a.auditable and a.kind == "choice" and a.self_consistent
    assert a.fragility_total >= 5 and a.fragility_flips == 0
    assert a.false_positive_total == 1 and a.false_positive_hits == 0
    assert a.is_boundary is False


def test_choice_integer_index_resolves_and_self_grades():
    # 0-based index 2 -> "C"; the canonical letter form must grade against its own reference.
    a = audit_item("gpqa", 2)
    assert a.auditable and a.self_consistent and a.fragility_flips == 0


def test_fraction_reference_skips_the_numeric_wrong_form():
    # No clean numeric value -> no false-positive probe (kept conservative, never invalid).
    a = audit_item("math500", "\\frac{1}{2}")
    assert a.auditable and a.self_consistent
    assert a.false_positive_total == 0
    assert a.is_boundary is False


# --------------------------------------------------------------------------- #
# the audit has teeth: it DETECTS a misbehaving grader (fake stand-in)
# --------------------------------------------------------------------------- #
def test_detects_fragility(monkeypatch):
    # Fragile grader: accepts a candidate carrying the gold token UNLESS it contains "$".
    def fake(bench, cand, ref):
        return 0.0 if "$" in cand else (1.0 if str(ref) in cand else 0.0)

    monkeypatch.setattr(grader_audit_pkg, "score_text", fake)
    a = audit_item("math500", "5")
    assert a.self_consistent                       # \boxed{5} carries "5", no "$"
    assert a.fragility_flips >= 1                   # the "$...$" perturbation trips it
    assert any(f.kind == "fragility" for f in a.findings)
    assert a.is_boundary


def test_detects_false_positive(monkeypatch):
    # Lax grader: accepts everything, so the clearly-wrong variant slips through.
    monkeypatch.setattr(grader_audit_pkg, "score_text", lambda bench, cand, ref: 1.0)
    a = audit_item("math500", "5")
    assert a.false_positive_total == 1 and a.false_positive_hits == 1
    assert any(f.kind == "false_positive" for f in a.findings)
    assert a.is_boundary


def test_self_consistency_failure_short_circuits_fragility(monkeypatch):
    # Broken grader: rejects everything, so even the canonical gold fails.
    monkeypatch.setattr(grader_audit_pkg, "score_text", lambda bench, cand, ref: 0.0)
    a = audit_item("math500", "5")
    assert a.self_consistent is False
    assert a.fragility_total == 0                   # not measured once the baseline is broken
    assert a.is_boundary


# --------------------------------------------------------------------------- #
# aggregation, rates, boundary sampling
# --------------------------------------------------------------------------- #
def test_aggregate_rates_and_skip_counts():
    items = [("math500", "5"), ("math500", "42"), ("mmlu", "A"), ("livecodebench", "stub")]
    res = audit(items, seed=0)
    by = {a.benchmark: a for a in res}
    assert by["math500"].n_auditable == 2 and by["math500"].estimated_error_rate == 0.0
    assert by["math500"].false_negative_rate == 0.0
    assert by["livecodebench"].n_skipped == 1 and by["livecodebench"].n_auditable == 0
    assert by["livecodebench"].estimated_error_rate is None


def test_boundary_sampling_is_capped_and_deterministic(monkeypatch):
    # Every item fails self-consistency -> every item is a boundary case.
    monkeypatch.setattr(grader_audit_pkg, "score_text", lambda b, c, r: 0.0)
    items = [("math500", str(i)) for i in range(50)]
    (m,) = audit(items, sample_size=10, seed=0)
    assert m.n_boundary == 50 and len(m.boundary_samples) == 10
    assert m.estimated_error_rate == 1.0
    # same seed -> same sample
    (m2,) = audit(items, sample_size=10, seed=0)
    assert [b.reference for b in m2.boundary_samples] == [b.reference for b in m.boundary_samples]


def test_render_has_table_rows_and_verdict():
    md = render(audit([("math500", "5"), ("mmlu", "B")], seed=0))
    assert "Grader audit" in md
    assert "| math500 |" in md and "| mmlu |" in md
    assert "Verdict:" in md


def test_render_flags_boundary_cases(monkeypatch):
    monkeypatch.setattr(grader_audit_pkg, "score_text", lambda b, c, r: 0.0)
    md = render(audit([("math500", "5")], seed=0))
    assert "boundary case" in md and "self_consistency" in md


# --------------------------------------------------------------------------- #
# report script end-to-end (protocol-item mapping via from_protocol_item)
# --------------------------------------------------------------------------- #
def test_report_script_reads_protocol_items(tmp_path):
    # Legacy on-disk field names (question_id / correct_answer) must map through.
    items = {"items": [
        {"question_id": "q1", "question_text": "?", "benchmark": "math500", "correct_answer": "5"},
        {"question_id": "q2", "question_text": "?", "benchmark": "mmlu", "correct_answer": "B"},
    ]}
    f = tmp_path / "items.json"
    f.write_text(json.dumps(items))
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "grader_audit_report.py"), str(f)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0, out.stderr
    assert "Grader audit" in out.stdout
    assert "| math500 |" in out.stdout and "| mmlu |" in out.stdout


def test_report_script_no_files_is_graceful():
    out = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "grader_audit_report.py")],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert out.returncode == 0
    assert "no item JSONs" in out.stdout
