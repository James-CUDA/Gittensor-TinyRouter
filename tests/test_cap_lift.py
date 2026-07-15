"""Offline tests for the R6 (token-cap lift -> beats all constituents) verifier.

No network, no GPU.
"""
from __future__ import annotations

import math

import pytest

from trinity.analysis.cap_lift import analyze_cap_lift, render


# ---------------------------------------------------------------------------
# The SPEC R6 example
# ---------------------------------------------------------------------------
def test_spec_example_holds():
    # docs/SPEC.md §1.3 R6: LiveCodeBench 0.61 -> 0.862 when the cap is lifted,
    # beating GPT-5 0.838, Gemini 0.672, Claude 0.465.
    r = analyze_cap_lift(
        {4096: 0.615, "uncapped": 0.862},
        {"gpt5": 0.838, "gemini": 0.672, "claude": 0.465},
        benchmark="livecodebench",
    )
    assert r.jump == pytest.approx(0.862 - 0.615)
    assert r.monotone
    assert r.beats_all_constituents
    assert r.best_constituent == "gpt5"
    assert r.margin == pytest.approx(0.862 - 0.838)
    assert r.holds


def test_uncapped_sorts_last_as_the_largest_budget():
    # +inf cap must sort after any finite cap regardless of input order.
    r = analyze_cap_lift([(None, 0.86), (4096, 0.61), (8192, 0.80)])
    assert r.caps[0] == 4096 and math.isinf(r.caps[-1])
    assert r.base_accuracy == 0.61 and r.lifted_accuracy == 0.86
    assert r.lifted_cap == math.inf


def test_lift_that_only_catches_up_does_not_hold():
    # Big jump, but the lifted accuracy still trails the best constituent (GPT-5):
    # the routed system only caught up, did not overtake -> R6 not shown.
    r = analyze_cap_lift(
        {4096: 0.61, "uncapped": 0.82},
        {"gpt5": 0.838, "gemini": 0.672},
    )
    assert r.jump == pytest.approx(0.21)
    assert not r.beats_all_constituents
    assert r.margin == pytest.approx(0.82 - 0.838)
    assert not r.holds


def test_a_drop_when_lifting_breaks_monotonicity():
    r = analyze_cap_lift({2048: 0.80, 4096: 0.90, "uncapped": 0.75},
                         {"gpt5": 0.70})
    assert not r.monotone
    assert r.max_drop == pytest.approx(0.15)          # 0.90 -> 0.75
    assert r.worst_step == (4096, math.inf)
    # net jump is lifted - base (0.75 - 0.80 = -0.05), so R6 does not hold
    assert r.jump == pytest.approx(-0.05) and not r.holds


def test_ties_with_a_constituent_do_not_count_as_beating_it():
    # strict domination: equalling GPT-5 is not beating it.
    r = analyze_cap_lift({4096: 0.61, "uncapped": 0.838}, {"gpt5": 0.838})
    assert not r.beats_all_constituents and not r.holds


def test_without_constituents_dominance_is_unproven():
    r = analyze_cap_lift({4096: 0.61, "uncapped": 0.862})
    assert r.jump > 0 and r.monotone
    assert not r.beats_all_constituents      # nothing to compare against
    assert not r.holds
    assert r.best_constituent is None


def test_single_cap_sweep_does_not_hold():
    r = analyze_cap_lift({4096: 0.9}, {"gpt5": 0.5})
    assert r.n_points == 1 and r.jump == 0.0 and not r.holds


def test_non_numeric_and_duplicate_entries_are_cleaned():
    # duplicate cap -> last wins; unparseable cap / non-numeric accuracy dropped.
    r = analyze_cap_lift([(4096, 0.6), (4096, 0.61), ("bogus", 0.7), (None, 0.86)])
    assert r.caps == [4096, math.inf]
    assert r.accuracies == [0.61, 0.86]


def test_string_inf_aliases_all_map_to_uncapped():
    for alias in ["inf", "Infinity", "UNCAPPED", "none", "off"]:
        r = analyze_cap_lift({4096: 0.6, alias: 0.9})
        assert math.isinf(r.lifted_cap) and r.lifted_accuracy == 0.9


def test_to_dict_labels_uncapped_and_is_json_shaped():
    d = analyze_cap_lift({4096: 0.61, "uncapped": 0.862},
                         {"gpt5": 0.838}).to_dict()
    assert d["caps"] == ["4096", "uncapped"]
    assert d["lifted_cap"] == "uncapped"
    assert d["holds"] is True
    assert d["best_constituent"] == "gpt5"


def test_render_reports_the_verdict():
    out = render({4096: 0.615, "uncapped": 0.862},
                 {"gpt5": 0.838, "gemini": 0.672, "claude": 0.465})
    assert "R6" in out and "HOLDS" in out and "livecodebench" in out
    out2 = render({4096: 0.61, "uncapped": 0.82}, {"gpt5": 0.838})
    assert "NOT SHOWN" in out2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
