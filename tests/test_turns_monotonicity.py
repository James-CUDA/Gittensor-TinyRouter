"""Offline tests for the R7 (turns -> monotonic gain) verifier. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.analysis.turns_monotonicity import (
    analyze_benchmarks,
    analyze_sweep,
    render,
)


# ---------------------------------------------------------------------------
# analyze_sweep
# ---------------------------------------------------------------------------
def test_monotone_increasing_sweep_holds():
    # The SPEC R7 example: 0.823 -> 0.863 over 2 -> 6 turns.
    s = analyze_sweep({2: 0.823, 4: 0.845, 6: 0.863}, benchmark="livecodebench")
    assert s.monotone and s.holds
    assert s.net_gain == pytest.approx(0.040)
    assert s.max_drop == 0.0 and s.worst_step is None


def test_accepts_unsorted_pairs_and_sorts_by_turns():
    s = analyze_sweep([(6, 0.863), (2, 0.823), (4, 0.845)])
    assert s.turns == [2, 4, 6]
    assert s.accuracies == [0.823, 0.845, 0.863]


def test_a_drop_breaks_monotonicity_and_records_the_worst_step():
    s = analyze_sweep({2: 0.80, 4: 0.90, 6: 0.75})
    assert not s.monotone and not s.holds
    assert s.max_drop == pytest.approx(0.15)   # 0.90 -> 0.75
    assert s.worst_step == (4, 6)
    # net gain is still last-minus-first (0.75 - 0.80 = -0.05), and negative.
    assert s.net_gain == pytest.approx(-0.05)


def test_flat_sweep_is_monotone_but_does_not_hold_r7():
    # No drop -> monotone, but zero gain -> R7 (which requires a GAIN) does not hold.
    s = analyze_sweep({2: 0.8, 4: 0.8, 6: 0.8})
    assert s.monotone and not s.holds and s.net_gain == 0.0


def test_float_noise_on_a_flat_step_is_not_a_violation():
    s = analyze_sweep({2: 0.800000000, 4: 0.7999999999}, tol=1e-6)
    assert s.monotone  # within tol
    # but net gain is ~0 (slightly negative), so R7 does not hold
    assert not s.holds


def test_single_point_sweep_does_not_hold():
    s = analyze_sweep({4: 0.9})
    assert s.n_points == 1 and not s.holds and s.net_gain == 0.0


def test_non_numeric_and_duplicate_entries_are_cleaned():
    # duplicate turn -> last wins; non-numeric dropped.
    s = analyze_sweep([(2, 0.8), (2, 0.82), (4, "oops"), (6, 0.9)])
    assert s.turns == [2, 6] and s.accuracies == [0.82, 0.9]


def test_roundtrips_to_dict():
    d = analyze_sweep({2: 0.8, 6: 0.9}, benchmark="mmlu").to_dict()
    assert d["benchmark"] == "mmlu" and d["holds"] is True
    assert d["worst_step"] is None


# ---------------------------------------------------------------------------
# analyze_benchmarks (union verdict)
# ---------------------------------------------------------------------------
def test_r7_holds_only_when_every_benchmark_holds():
    good = analyze_benchmarks({
        "math500": {2: 0.5, 6: 0.6},
        "mmlu": {2: 0.7, 6: 0.72},
    })
    assert good["r7_holds"] is True
    assert good["union_net_gain"] == pytest.approx((0.1 + 0.02) / 2)
    assert good["violations"] == []

    bad = analyze_benchmarks({
        "math500": {2: 0.5, 6: 0.6},
        "livecodebench": {2: 0.8, 6: 0.7},   # regresses
    })
    assert bad["r7_holds"] is False
    assert bad["violations"] == ["livecodebench"]


def test_empty_input_does_not_hold():
    report = analyze_benchmarks({})
    assert report["r7_holds"] is False and report["union_net_gain"] == 0.0


def test_render_reports_the_verdict():
    out = render({"math500": {2: 0.5, 4: 0.55, 6: 0.6}})
    assert "R7" in out and "HOLDS" in out
    out2 = render({"math500": {2: 0.6, 6: 0.5}})
    assert "VIOLATED" in out2 and "math500" in out2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
