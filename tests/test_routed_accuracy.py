"""Offline tests for the routed-accuracy diagnostic. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.analysis.routed_accuracy import analyze, render


def test_shares_accuracy_and_contribution():
    # A routed 3x (2 right), B routed 1x (1 right) -> overall 3/4.
    recs = [("A", True), ("A", True), ("A", False), ("B", True)]
    s = analyze(recs)
    assert s.n_decisions == 4
    assert s.overall_accuracy == pytest.approx(0.75)
    by = {m.model: m for m in s.per_model}
    assert by["A"].n_routed == 3 and by["A"].share == pytest.approx(0.75)
    assert by["A"].routed_accuracy == pytest.approx(2 / 3)
    assert by["A"].contribution == pytest.approx(2 / 4)
    # contributions sum to overall accuracy.
    assert sum(m.contribution for m in s.per_model) == pytest.approx(s.overall_accuracy)


def test_best_and_worst_routed_model():
    recs = [("A", True), ("A", True), ("B", False), ("B", False)]
    s = analyze(recs)
    assert s.best_model == "A" and s.worst_model == "B"


def test_overused_model_is_high_share_below_average():
    # A: big share but 0.5 acc (below overall 0.6); B: small share, perfect.
    recs = [("A", True), ("A", False)] * 4 + [("B", True), ("B", True)]
    s = analyze(recs)
    # overall = (4 + 2) / 10 = 0.6 ; A acc 0.5 < 0.6 and has the largest share.
    assert s.overall_accuracy == pytest.approx(0.6)
    assert s.overused_model == "A"


def test_no_overused_model_when_all_at_or_above_overall():
    recs = [("A", True), ("B", True), ("C", True)]
    s = analyze(recs)
    assert s.overall_accuracy == pytest.approx(1.0)
    assert s.overused_model is None


def test_correct_accepts_bool_int_and_score_and_string():
    recs = [
        {"model": "A", "correct": True},
        {"model": "A", "is_correct": 1},
        {"model": "A", "score": 0.0},       # score 0 -> wrong
        {"model": "A", "correct": "pass"},
    ]
    s = analyze(recs)
    a = s.per_model[0]
    assert a.n_routed == 4 and a.routed_accuracy == pytest.approx(0.75)


def test_unusable_records_are_skipped():
    recs = [("A", None), ("", True), None, ("A", 1)]
    s = analyze(recs)
    assert s.n_decisions == 1 and s.overall_accuracy == pytest.approx(1.0)


def test_shares_sum_to_one():
    recs = [("A", True), ("B", False), ("C", True), ("A", False)]
    s = analyze(recs)
    assert sum(m.share for m in s.per_model) == pytest.approx(1.0)
    # sorted by share desc (A has 2, others 1).
    assert s.per_model[0].model == "A"


def test_empty_input_is_zeroed():
    s = analyze([])
    assert s.n_decisions == 0 and s.overall_accuracy == 0.0
    assert s.per_model == [] and s.best_model is None and s.overused_model is None


def test_render_reports_overall_and_overuse():
    recs = [("A", True), ("A", False)] * 4 + [("B", True), ("B", True)]
    md = render(recs)
    assert "overall routed accuracy: 0.600" in md
    assert "over-used relative to its routed accuracy: A" in md
    assert "| model | routed | share | routed acc | contribution |" in md


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
