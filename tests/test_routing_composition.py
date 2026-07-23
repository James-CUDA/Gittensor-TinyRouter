"""Offline tests for the routing-composition diagnostic. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.analysis.routing_composition import (
    analyze,
    analyze_benchmarks,
    render,
)


# ---------------------------------------------------------------------------
# analyze — shares, unused, entropy, collapse
# ---------------------------------------------------------------------------
def test_shares_sum_to_one_and_count_decisions():
    decs = [("A", "worker"), ("A", "thinker"), ("B", "verifier"), ("B", "worker")]
    r = analyze(decs, benchmark="math500")
    assert r.n_decisions == 4
    assert r.model_shares == {"A": 0.5, "B": 0.5}
    assert sum(r.role_shares.values()) == pytest.approx(1.0)
    assert r.role_shares["worker"] == pytest.approx(0.5)


def test_unused_pool_models_and_roles_are_reported():
    decs = [("A", "worker"), ("A", "worker"), ("A", "thinker")]
    r = analyze(decs, pool_models=["A", "B", "C"])
    assert r.unused_models == ["B", "C"]
    # only thinker + worker seen -> verifier never used.
    assert r.unused_roles == ["verifier"]


def test_collapse_flag_when_one_model_dominates():
    decs = [("A", "worker")] * 19 + [("B", "worker")]   # A = 0.95 >= 0.90
    r = analyze(decs)
    assert r.top_model == "A" and r.top_model_share == pytest.approx(0.95)
    assert r.collapsed is True


def test_not_collapsed_when_spread_below_threshold():
    decs = [("A", "worker")] * 8 + [("B", "worker")] * 2   # A = 0.8 < 0.9
    assert analyze(decs).collapsed is False


def test_entropy_is_zero_for_single_model_and_one_for_uniform():
    assert analyze([("A", "worker")] * 5).model_entropy == pytest.approx(0.0)
    uniform = analyze([("A", "w"), ("B", "w"), ("C", "w")]).model_entropy
    assert uniform == pytest.approx(1.0)


def test_a_single_model_run_is_collapsed():
    r = analyze([("only", "worker")] * 10, pool_models=["only", "other"])
    assert r.collapsed is True and r.unused_models == ["other"]


def test_accepts_mapping_and_bare_model_and_role_enum_value_shapes():
    class _Role:
        value = "Verifier"
    decs = [{"model": "A", "role": "Worker"}, "B", ("C", _Role())]
    r = analyze(decs)
    assert r.n_decisions == 3
    # roles are lower-cased; the bare model "B" contributes no role.
    assert r.role_shares == {"worker": 0.5, "verifier": 0.5}


def test_empty_and_malformed_entries_are_skipped():
    r = analyze([(), ("", "worker"), None, ("A", "worker")])
    assert r.n_decisions == 1 and r.model_shares == {"A": 1.0}


# ---------------------------------------------------------------------------
# analyze_benchmarks — per-benchmark + pooled union
# ---------------------------------------------------------------------------
def test_union_pools_all_decisions_and_flags_collapsed_benchmarks():
    per = {
        "math500": [("A", "worker")] * 10,                 # collapsed (all A)
        "mmlu": [("A", "worker")] * 5 + [("B", "verifier")] * 5,  # not collapsed
    }
    report = analyze_benchmarks(per, pool_models=["A", "B"])
    assert report["union"]["n_decisions"] == 20
    assert report["any_collapsed"] is True
    assert report["collapsed_benchmarks"] == ["math500"]
    # union share of A = 15/20.
    assert report["union"]["model_shares"]["A"] == pytest.approx(0.75)


def test_render_reports_collapse_and_unused_roles():
    per = {"math500": [("A", "worker")] * 10}
    md = render(per, pool_models=["A", "B"])
    assert "collapsed (one model >= threshold) on: math500" in md
    assert "roles never used (union): thinker, verifier" in md
    assert "| union |" in md


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
