"""Offline tests for the R5 (TRINITY ≈ Per-Question-Best) verifier. No network, no GPU."""
from __future__ import annotations

import math

import pytest

from trinity.analysis.per_question_best import (
    analyze_benchmarks,
    analyze_task,
    render,
)


# ---------------------------------------------------------------------------
# analyze_task
# ---------------------------------------------------------------------------
def test_reaches_ceiling_within_tolerance():
    # The SPEC R5 shape: TRINITY 0.845 vs a 0.862 per-question ceiling -> within 5pts.
    g = analyze_task("livecodebench", 0.845, 0.862)
    assert g.gap == pytest.approx(0.017)
    assert g.reaches is True


def test_short_of_ceiling_beyond_tolerance():
    g = analyze_task("math500", 0.50, 0.62)  # 12-point gap
    assert g.gap == pytest.approx(0.12)
    assert g.reaches is False


def test_trinity_at_or_above_ceiling_is_negative_gap_and_reaches():
    g = analyze_task("mmlu", 0.71, 0.70)  # noisy estimate: TRINITY slightly above
    assert g.gap == pytest.approx(-0.01)
    assert g.reaches is True


def test_fraction_closed_uses_baseline_headroom():
    # baseline 0.60, ceiling 0.90, trinity 0.84 -> captured (0.84-0.60)/(0.90-0.60)=0.80
    g = analyze_task("lcb", 0.84, 0.90, baseline=0.60)
    assert g.fraction_closed == pytest.approx(0.80)


def test_fraction_closed_is_nan_without_headroom_or_baseline():
    assert math.isnan(analyze_task("x", 0.5, 0.5).fraction_closed)          # no baseline
    assert math.isnan(analyze_task("x", 0.5, 0.5, baseline=0.5).fraction_closed)  # no headroom


def test_custom_tolerance():
    assert analyze_task("x", 0.80, 0.83, tol=0.02).reaches is False
    assert analyze_task("x", 0.80, 0.83, tol=0.05).reaches is True


# ---------------------------------------------------------------------------
# analyze_benchmarks + the R5 verdict
# ---------------------------------------------------------------------------
def test_r5_holds_when_all_but_one_reach_the_ceiling():
    # 3 benchmarks, default min_pass = n-1 = 2. Two reach, one is short -> holds.
    tasks = {
        "livecodebench": {"trinity": 0.845, "per_question_best": 0.862},
        "mmlu": {"trinity": 0.70, "per_question_best": 0.71},
        "math500": {"trinity": 0.50, "per_question_best": 0.62},  # short
    }
    report = analyze_benchmarks(tasks)
    assert report["n_scored"] == 3 and report["n_reached"] == 2
    assert report["min_pass"] == 2
    assert report["r5_holds"] is True
    assert report["misses"] == ["math500"]


def test_r5_violated_when_too_many_fall_short():
    tasks = {
        "a": {"trinity": 0.40, "per_question_best": 0.80},  # short
        "b": {"trinity": 0.50, "per_question_best": 0.90},  # short
        "c": {"trinity": 0.70, "per_question_best": 0.71},  # reaches
    }
    report = analyze_benchmarks(tasks)  # min_pass=2, only 1 reaches
    assert report["r5_holds"] is False
    assert report["n_reached"] == 1

    # union mean gap is the equal-weight average of the three gaps.
    assert report["union_mean_gap"] == pytest.approx((0.40 + 0.40 + 0.01) / 3)


def test_explicit_min_pass_overrides_default():
    tasks = {
        "a": {"trinity": 0.845, "per_question_best": 0.862},
        "b": {"trinity": 0.50, "per_question_best": 0.62},
    }
    assert analyze_benchmarks(tasks, min_pass=2)["r5_holds"] is False  # needs both
    assert analyze_benchmarks(tasks, min_pass=1)["r5_holds"] is True   # one is enough


def test_bare_number_entry_is_treated_as_pqb_and_skipped_without_trinity():
    # A benchmark carrying only the PQB number (no TRINITY) has nothing to compare.
    report = analyze_benchmarks({"a": 0.80})
    assert report["n_scored"] == 0 and report["r5_holds"] is False


def test_non_numeric_or_missing_values_are_skipped():
    tasks = {
        "ok": {"trinity": 0.70, "per_question_best": 0.72},
        "no_pqb": {"trinity": 0.70},
        "bad": {"trinity": "x", "per_question_best": 0.7},
    }
    report = analyze_benchmarks(tasks)
    assert [r["benchmark"] for r in report["per_benchmark"]] == ["ok"]


def test_accepts_short_key_aliases():
    report = analyze_benchmarks({"a": {"trinity_accuracy": 0.70, "pqb": 0.72}})
    assert report["n_scored"] == 1 and report["per_benchmark"][0]["reaches"] is True


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def test_render_reports_verdict_and_misses():
    tasks = {
        "livecodebench": {"trinity": 0.845, "per_question_best": 0.862},
        "math500": {"trinity": 0.50, "per_question_best": 0.62},
    }
    md = render(tasks)  # min_pass = n-1 = 1, one reaches -> HOLDS
    assert "R5 (TRINITY ≈ Per-Question-Best): HOLDS" in md
    assert "reached the ceiling on 1/2 benchmarks" in md
    assert "short of ceiling: math500" in md


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
