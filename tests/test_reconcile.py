"""Offline tests for the oracle-matrix <-> rigorous-eval reconciliation (§5.3 integrity guard).

Synthetic matrices in the ``oracle_matrix`` schema + a real-data check on the committed
math500/mmlu artifacts (which must reconcile — verdict TRUSTWORTHY). numpy+stdlib only,
no torch/scipy/network.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis import reconcile as reconcile_pkg  # re-export check (function)
from trinity.analysis.reconcile import (
    TRUSTWORTHY,
    VOID,
    matrix_accuracy,
    reconcile,
    reconcile_files,
    render,
    rigorous_accuracy,
)

_REPO = Path(__file__).resolve().parents[1]
_FINAL = _REPO / "experiments" / "final"


def _matrix(per_model_by_q, benchmark="math500"):
    """per_model_by_q: list of {model: [0/1,...K]} dicts (one per question)."""
    return {"benchmark": benchmark,
            "tasks": [{"id": f"q{i}", "per_model": pm} for i, pm in enumerate(per_model_by_q)]}


def _rig(accs, stds, benchmark="math500"):
    """Build a <bench>_rigorous.json-shaped dict from per-model acc/std maps."""
    results = {}
    for m, a in accs.items():
        results[f"single::{m}"] = a
        results[f"single_std::{m}"] = stds[m]
    return {"benchmark": benchmark, "results": results}


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.analysis.reconcile; "
            "assert 'torch' not in sys.modules and 'scipy' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


def test_reexported_from_package():
    assert reconcile_pkg is reconcile


# --------------------------------------------------------------------------- #
# helpers: matrix_accuracy + rigorous_accuracy
# --------------------------------------------------------------------------- #
def test_matrix_accuracy_mean_and_se():
    # a solves 4/5 on q0 and 2/5 on q1 -> per-query rates .8, .4 ; mean .6.
    acc, se, k, n, models = matrix_accuracy(
        _matrix([{"a": [1, 1, 1, 1, 0], "b": [0, 0, 0, 0, 0]},
                 {"a": [1, 1, 0, 0, 0], "b": [0, 0, 0, 0, 0]}]))
    assert k == 5 and n == 2 and models == ["a", "b"]
    assert acc["a"] == pytest.approx(0.6) and acc["b"] == pytest.approx(0.0)
    assert se["a"] > 0.0 and se["b"] == pytest.approx(0.0)   # b is constant -> zero SE


def test_rigorous_accuracy_parses_single_and_std():
    acc, std, models = rigorous_accuracy(
        _rig({"a": 0.75, "b": 0.20}, {"a": 0.02, "b": 0.03}))
    assert acc == {"a": 0.75, "b": 0.20}
    assert std == {"a": 0.02, "b": 0.03} and set(models) == {"a", "b"}


def test_rigorous_accuracy_missing_std_defaults_zero():
    acc, std, _ = rigorous_accuracy({"results": {"single::a": 0.5}})
    assert acc == {"a": 0.5} and std == {"a": 0.0}


# --------------------------------------------------------------------------- #
# reconcile verdicts
# --------------------------------------------------------------------------- #
def test_matching_matrix_is_trustworthy():
    matrix = _matrix([{"a": [1, 1, 1, 1, 0], "b": [1, 0, 0, 0, 0]}] * 10)  # a=.8, b=.2
    rig = _rig({"a": 0.80, "b": 0.20}, {"a": 0.02, "b": 0.02})
    s = reconcile(matrix, rig)
    assert s.verdict == TRUSTWORTHY and s.problems == []
    assert s.models_match and s.best_model_agrees and s.rank_agrees
    assert all(p.status == "reconciled" for p in s.per_model)


def test_shifted_matrix_is_void():
    # matrix says a=.8 (constant -> zero SE); rigorous says .5 +-.01 -> z=30.
    matrix = _matrix([{"a": [1, 1, 1, 1, 0], "b": [1, 0, 0, 0, 0]}] * 10)
    rig = _rig({"a": 0.50, "b": 0.20}, {"a": 0.01, "b": 0.02})
    s = reconcile(matrix, rig)
    assert s.verdict == VOID
    a = next(p for p in s.per_model if p.model == "a")
    assert a.status == "void" and a.z > 3.0
    assert any("a:" in p for p in s.problems)


def test_model_set_mismatch_is_suspect_and_flagged():
    matrix = _matrix([{"a": [1, 1, 1, 1, 0], "b": [1, 0, 0, 0, 0]}] * 10)  # a, b
    rig = _rig({"a": 0.80, "c": 0.30}, {"a": 0.02, "c": 0.02})              # a, c
    s = reconcile(matrix, rig)
    assert s.matrix_only == ["b"] and s.rigorous_only == ["c"]
    assert not s.models_match and s.verdict == "SUSPECT"
    assert s.models == ["a"]                          # only the shared model is scored
    assert any("matrix but not rigorous" in p for p in s.problems)


def test_best_model_disagreement_is_suspect():
    # both models reconcile (wide rigorous std) but the matrix's best is the rigorous
    # worst -> a best-single claim could differ, so escalate to SUSPECT.
    a20 = [1] * 16 + [0] * 4      # 16/20 = .80
    b20 = [1] * 15 + [0] * 5      # 15/20 = .75
    matrix = _matrix([{"a": a20, "b": b20}] * 8)
    rig = _rig({"a": 0.75, "b": 0.80}, {"a": 0.10, "b": 0.10})
    s = reconcile(matrix, rig)
    assert s.best_model_matrix == "a" and s.best_model_rigorous == "b"
    assert not s.best_model_agrees and s.verdict == "SUSPECT"
    assert all(p.status == "reconciled" for p in s.per_model)   # values individually agree


def test_no_shared_models_is_void():
    s = reconcile(_matrix([{"a": [1, 0]}] * 4), _rig({"z": 0.5}, {"z": 0.01}))
    assert s.verdict == VOID and s.models == []
    assert any("no shared models" in p for p in s.problems)


def test_ragged_matrix_raises():
    with pytest.raises(ValueError):
        reconcile(_matrix([{"a": [1, 1, 0]}, {"a": [1, 0]}]), _rig({"a": 0.5}, {"a": 0.01}))


# --------------------------------------------------------------------------- #
# real committed data: the matrix reproduces the rigorous eval (TRUSTWORTHY)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("matrix_name,rig_name", [
    ("oracle_matrix_math500.json", "math_rigorous.json"),
    ("oracle_matrix_mmlu.json", "mmlu_rigorous.json"),
])
def test_real_artifacts_reconcile(matrix_name, rig_name):
    mp, rp = _FINAL / matrix_name, _FINAL / rig_name
    if not (mp.exists() and rp.exists()):
        pytest.skip("committed oracle_matrix / rigorous artifacts not present")
    s = reconcile_files(mp, rp)
    assert s.n == 120 and len(s.models) == 3
    assert s.verdict == TRUSTWORTHY and s.problems == []
    assert s.models_match and s.best_model_agrees and s.rank_agrees
    assert all(abs(p.z) < 2.0 for p in s.per_model)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_report():
    matrix = _matrix([{"a": [1, 1, 1, 1, 0], "b": [1, 0, 0, 0, 0]}] * 10)
    md = render(reconcile(matrix, _rig({"a": 0.80, "b": 0.20}, {"a": 0.02, "b": 0.02})))
    assert "reconciliation" in md.lower() and "TRUSTWORTHY" in md
    assert "combined" in md and "| a |" in md
    empty = render(reconcile({"benchmark": "x", "tasks": []}, {"results": {}}))
    assert empty.strip().endswith("(no data to reconcile)_")
