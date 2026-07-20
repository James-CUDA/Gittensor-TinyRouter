"""Offline tests for the cross-benchmark (equal-weight union) sampling/selective view.

The competition score is the equal-weighted union of the benchmarks, so the headline
questions — *does re-sampling the best single model rival the routing oracle?* and *is
self-consistency confidence informative enough to abstain on?* — have to be answered on the
union, not one benchmark at a time. The case that matters is a **split**: a model that rivals
the ceiling on one task and not the other. These tests pin that the union answer is the
composite's answer, that the split is surfaced rather than averaged away, and that the union
never re-derives the per-benchmark math it aggregates. Synthetic matrices, numpy only.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis import union_diagnostics as union_diagnostics_pkg  # re-export check
from trinity.analysis.sampling import analyze as analyze_sampling
from trinity.analysis.selective import analyze as analyze_selective
from trinity.analysis.union_diagnostics import (
    render_sampling,
    render_selective,
    union_sampling,
    union_selective,
)

_REPO = Path(__file__).resolve().parents[1]


def _matrix(benchmark: str, solves: dict[str, list[int]], k: int = 5) -> dict:
    """``{model: [n_correct_of_k per query]}`` -> an ``oracle_matrix`` dict."""
    n = len(next(iter(solves.values())))
    tasks = [
        {"id": f"q{i}", "per_model": {m: [1] * solves[m][i] + [0] * (k - solves[m][i])
                                      for m in solves}}
        for i in range(n)
    ]
    return {"benchmark": benchmark, "k": k, "tasks": tasks}


# A dominates on math500; B dominates on mmlu. Complementary -> real routing headroom.
_MATH = _matrix("math500", {"alpha": [5, 5, 4, 5, 0, 5], "beta": [0, 1, 5, 0, 5, 1]})
_MMLU = _matrix("mmlu", {"alpha": [0, 1, 0, 1, 5, 0], "beta": [5, 4, 5, 5, 0, 5]})
# Both models solve everything -> majority trivially rivals the (equal) oracle.
_EASY = _matrix("gpqa", {"alpha": [5, 5, 5, 5], "beta": [5, 5, 5, 5]})


# --------------------------------------------------------------------------- #
# sampling union
# --------------------------------------------------------------------------- #
def test_union_is_the_equal_weight_mean_of_the_per_benchmark_analyses():
    """The union must not re-derive the math — it averages the canonical per-bench values."""
    u = union_sampling([_MATH, _MMLU])
    a, b = analyze_sampling(_MATH), analyze_sampling(_MMLU)
    got = {p.model: p.pass_at_1 for p in u.per_model}
    for m in u.models:
        pa = next(p.pass_at_1 for p in a.per_model if p.model == m)
        pb = next(p.pass_at_1 for p in b.per_model if p.model == m)
        assert got[m] == pytest.approx((pa + pb) / 2)
    assert u.routing_oracle == pytest.approx((a.routing_oracle + b.routing_oracle) / 2)
    assert u.n_benchmarks == 2


def test_complementary_models_leave_routing_headroom_on_the_union():
    u = union_sampling([_MATH, _MMLU])
    assert u.majority_rivals_oracle is False
    assert u.best_majority < u.routing_oracle
    assert "real headroom" in render_sampling(u)


def test_union_surfaces_a_split_verdict_instead_of_averaging_it_away():
    """The motivating case: benchmarks disagree, so the split must be reported."""
    u = union_sampling([_MATH, _EASY])
    assert set(u.per_benchmark_rivals) == {"math500", "gpqa"}
    assert u.per_benchmark_rivals["gpqa"] is True        # everyone solves everything
    assert u.per_benchmark_rivals["math500"] is False    # complementary specialists
    assert u.verdict_is_unanimous is False
    assert "DISAGREE" in render_sampling(u)


def test_unanimous_verdict_is_not_flagged_as_split():
    u = union_sampling([_MATH, _MMLU])
    assert u.verdict_is_unanimous is True
    assert "DISAGREE" not in render_sampling(u)


def test_best_majority_is_picked_on_the_union_not_per_benchmark():
    # alpha wins math500, beta wins mmlu; the union winner is whichever leads on average.
    u = union_sampling([_MATH, _MMLU])
    best = max(u.per_model, key=lambda p: p.majority_at_k)
    assert u.best_majority_model == best.model
    assert u.best_majority == pytest.approx(best.majority_at_k)


# --------------------------------------------------------------------------- #
# selective union
# --------------------------------------------------------------------------- #
def test_selective_union_averages_and_picks_lowest_aurc():
    u = union_selective([_MATH, _MMLU])
    a, b = analyze_selective(_MATH), analyze_selective(_MMLU)
    for m in u.models:
        pa = next(p.aurc for p in a.per_model if p.model == m)
        pb = next(p.aurc for p in b.per_model if p.model == m)
        got = next(p.aurc for p in u.per_model if p.model == m)
        assert got == pytest.approx((pa + pb) / 2)
    assert u.best_aurc == pytest.approx(min(p.aurc for p in u.per_model))  # lower is better


def test_selective_reports_where_confidence_is_uninformative():
    u = union_selective([_MATH, _MMLU])
    assert set(u.per_benchmark_informative) == {"math500", "mmlu"}
    if u.any_confidence_informative and not u.informative_everywhere:
        assert "uninformative on" in render_selective(u)


def test_selective_informative_everywhere_flag():
    u = union_selective([_MATH, _MMLU])
    assert u.informative_everywhere == all(u.per_benchmark_informative.values())


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_mismatched_model_sets_raise_like_union_oracle():
    other = _matrix("gpqa", {"alpha": [5, 0], "gamma": [0, 5]})
    with pytest.raises(ValueError, match="expected"):
        union_sampling([_MATH, other])
    with pytest.raises(ValueError, match="expected"):
        union_selective([_MATH, other])


def test_empty_and_questionless_inputs_are_handled():
    for fn, render in ((union_sampling, render_sampling), (union_selective, render_selective)):
        empty = fn([])
        assert empty.n_benchmarks == 0 and empty.per_model == []
        assert "no benchmark matrices" in render(empty)
        # a matrix with no tasks contributes nothing but is still listed
        blank = fn([{"benchmark": "blank", "k": 5, "tasks": []}])
        assert blank.n_benchmarks == 0


def test_single_benchmark_union_equals_that_benchmark():
    u = union_sampling([_MATH])
    a = analyze_sampling(_MATH)
    assert u.n_benchmarks == 1
    assert u.routing_oracle == pytest.approx(a.routing_oracle)
    assert u.majority_rivals_oracle == a.majority_rivals_oracle


def test_summaries_are_json_serializable():
    json.dumps(union_sampling([_MATH, _MMLU]).to_dict())
    json.dumps(union_selective([_MATH, _MMLU]).to_dict())


def test_module_is_reachable_through_the_analysis_package():
    assert hasattr(union_diagnostics_pkg, "union_sampling")
    assert hasattr(union_diagnostics_pkg, "union_selective")


# --------------------------------------------------------------------------- #
# report script
# --------------------------------------------------------------------------- #
def _run(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "union_diagnostics_report.py"), *args],
        capture_output=True, text=True, cwd=str(cwd or _REPO),
    )


def test_report_script_renders_both_sections(tmp_path):
    for m in (_MATH, _MMLU):
        (tmp_path / f"oracle_matrix_{m['benchmark']}.json").write_text(json.dumps(m))
    out = _run(["--root", str(tmp_path)])
    assert out.returncode == 0, out.stderr
    assert "Cross-benchmark sampling" in out.stdout
    assert "Cross-benchmark selective prediction" in out.stdout


def test_report_script_only_flag_and_json(tmp_path):
    for m in (_MATH, _MMLU):
        (tmp_path / f"oracle_matrix_{m['benchmark']}.json").write_text(json.dumps(m))
    js = tmp_path / "out.json"
    out = _run(["--root", str(tmp_path), "--only", "sampling", "--json", str(js)])
    assert out.returncode == 0
    assert "Cross-benchmark sampling" in out.stdout
    assert "selective prediction" not in out.stdout
    assert set(json.loads(js.read_text())) == {"sampling"}


def test_report_script_explains_a_model_set_mismatch(tmp_path):
    (tmp_path / "oracle_matrix_math500.json").write_text(json.dumps(_MATH))
    (tmp_path / "oracle_matrix_gpqa.json").write_text(
        json.dumps(_matrix("gpqa", {"alpha": [5, 0], "gamma": [0, 5]})))
    out = _run(["--root", str(tmp_path)])
    assert out.returncode == 0                      # explains, does not traceback
    assert "cannot form the union" in out.stdout


def test_report_script_no_files_is_graceful():
    out = _run([])
    assert out.returncode == 0 and "no oracle_matrix JSONs" in out.stdout
