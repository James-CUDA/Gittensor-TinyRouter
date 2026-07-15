"""Band-boundary + serialization coverage for the matrix<->rigorous reconciliation.

``tests/test_reconcile.py`` exercises the TRUSTWORTHY happy path and the far-VOID
(``|z| >> 3``) shift, but three decision paths in
``trinity.analysis.reconcile`` carry no test:

* the **per-model SUSPECT band** — ``z_ok < |z| <= z_void`` — the middle verdict
  between reconciled and void;
* the **zero-combined-sigma** branch, where both artifacts claim zero uncertainty
  so any nonzero gap is a hard contradiction (``z = +-inf`` -> VOID) while an exact
  match stays ``z = 0`` (reconciled);
* ``rigorous_accuracy`` skipping a non-string key, the ``to_dict`` views, and the
  ``render`` "Problems:" section.

These are pure numpy/stdlib checks — no torch, scipy, or network.
"""
import json
import math

from trinity.analysis.reconcile import (
    SUSPECT,
    TRUSTWORTHY,
    VOID,
    PerModelReconciliation,
    ReconciliationSummary,
    reconcile,
    rigorous_accuracy,
    render,
)


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


# --------------------------------------------------------------------------- #
# per-model SUSPECT band: z_ok < |z| <= z_void
# --------------------------------------------------------------------------- #
def test_per_model_suspect_band():
    # a single-question matrix takes matrix_accuracy's n==1 branch, so matrix_se is
    # exactly 0 and sigma == the rigorous std (.02); z = (.80 - .75) / .02 = 2.5,
    # which lands strictly inside the suspect band (z_ok=2, z_void=3].
    matrix = _matrix([{"a": [1, 1, 1, 1, 0]}])              # .80, n=1 -> se 0
    s = reconcile(matrix, _rig({"a": 0.75}, {"a": 0.02}))

    (a,) = s.per_model
    assert a.matrix_se == 0.0                                # constant column
    assert 2.0 < abs(a.z) <= 3.0
    assert a.status == "suspect"
    assert s.verdict == SUSPECT
    assert s.models_match and s.best_model_agrees            # nothing else escalates
    assert any(p.startswith("a:") and "suspect" in p for p in s.problems)


# --------------------------------------------------------------------------- #
# zero combined sigma: exact match reconciles, any gap is a VOID contradiction
# --------------------------------------------------------------------------- #
def test_zero_sigma_exact_match_reconciles():
    # both sides claim zero uncertainty AND agree exactly -> z == 0, reconciled.
    # n=1 gives an exactly-zero matrix SE (not float noise), so sigma is exactly 0.
    matrix = _matrix([{"a": [1, 1, 1, 1, 0]}])              # .80, n=1 -> se 0
    s = reconcile(matrix, _rig({"a": 0.80}, {"a": 0.0}))    # rigorous std 0

    (a,) = s.per_model
    assert a.combined_sigma == 0.0
    assert a.z == 0.0 and a.status == "reconciled"
    assert s.verdict == TRUSTWORTHY and s.problems == []


def test_zero_sigma_mismatch_is_void_contradiction():
    # zero uncertainty on both sides but the means differ -> infinite z -> VOID.
    matrix = _matrix([{"a": [1, 1, 1, 1, 0]}])              # .80, n=1 -> se 0
    s = reconcile(matrix, _rig({"a": 0.60}, {"a": 0.0}))    # rigorous std 0, acc .60

    (a,) = s.per_model
    assert a.combined_sigma == 0.0
    assert math.isinf(a.z) and a.z > 0                       # copysign(inf, +diff)
    assert a.status == "void" and s.verdict == VOID


# --------------------------------------------------------------------------- #
# rigorous_accuracy: a non-string key is skipped, not parsed
# --------------------------------------------------------------------------- #
def test_rigorous_accuracy_skips_non_string_key():
    acc, std, models = rigorous_accuracy(
        {"results": {123: 0.9, "single::a": 0.8, "single_std::a": 0.01}})
    assert acc == {"a": 0.8} and std == {"a": 0.01} and models == ["a"]


# --------------------------------------------------------------------------- #
# to_dict views round-trip through JSON
# --------------------------------------------------------------------------- #
def test_to_dict_views_are_json_serializable():
    matrix = _matrix([{"a": [1, 1, 1, 1, 0], "b": [1, 0, 0, 0, 0]}] * 10)
    s = reconcile(matrix, _rig({"a": 0.80, "b": 0.20}, {"a": 0.02, "b": 0.02}))

    summary_d = s.to_dict()
    assert summary_d["verdict"] == TRUSTWORTHY
    assert summary_d["k"] == 5 and summary_d["n"] == 10
    assert {p["model"] for p in summary_d["per_model"]} == {"a", "b"}
    # every PerModelReconciliation.to_dict is a flat serializable record
    pm = s.per_model[0]
    assert isinstance(pm, PerModelReconciliation)
    assert set(pm.to_dict()) == {
        "model", "matrix_acc", "matrix_se", "rigorous_acc",
        "rigorous_std", "combined_sigma", "z", "status",
    }
    assert isinstance(s, ReconciliationSummary)
    json.dumps(summary_d)                                    # must not raise


# --------------------------------------------------------------------------- #
# render surfaces the Problems: section when a model drifts
# --------------------------------------------------------------------------- #
def test_render_lists_problems_on_void():
    matrix = _matrix([{"a": [1, 1, 1, 1, 0]}])              # .80, n=1 -> se 0
    md = render(reconcile(matrix, _rig({"a": 0.60}, {"a": 0.0})))
    assert "VOID" in md
    assert "Problems:" in md
    assert any(line.strip().startswith("- a:") for line in md.splitlines())
