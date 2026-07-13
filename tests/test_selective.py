"""Offline tests for the per-model selective-prediction / risk-coverage diagnostic.

Synthetic K-sample oracle matrices (where confidence is/ isn't informative) + a real-data
check on the committed math500 matrix. numpy-only, no torch/network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity.analysis import analyze_selective  # re-export check
from trinity.analysis.selective import analyze, render, risk_coverage

_REPO = Path(__file__).resolve().parents[1]


def _matrix(per_model_by_q, benchmark="math500"):
    """per_model_by_q: list of {model: [0/1,...K]} dicts (one per question)."""
    return {"benchmark": benchmark,
            "tasks": [{"id": f"q{i}", "per_model": pm} for i, pm in enumerate(per_model_by_q)]}


def test_module_imports_without_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.analysis.selective; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


def test_reexported_from_package():
    assert analyze_selective is analyze


# --------------------------------------------------------------------------- #
# risk_coverage curve mechanics
# --------------------------------------------------------------------------- #
def test_risk_coverage_orders_by_confidence():
    # q0 low-confidence wrong, q1 high-confidence right: at coverage 1/2 only q1 is
    # answered (higher confidence) -> selective accuracy 1.0, then drops to 0.5 at full.
    correct = np.array([0, 1])
    confidence = np.array([0.5, 1.0])
    coverage, sel_acc = risk_coverage(correct, confidence)
    assert coverage.tolist() == [0.5, 1.0]
    assert sel_acc.tolist() == [1.0, 0.5]


def test_risk_coverage_empty():
    cov, acc = risk_coverage(np.zeros(0), np.zeros(0))
    assert cov.size == 0 and acc.size == 0


# --------------------------------------------------------------------------- #
# analyze — confidence informative vs not
# --------------------------------------------------------------------------- #
def test_confidence_informative_when_agreement_tracks_correctness():
    # a: on 2 queries the K samples fully agree AND are right (5/5), on 2 they split (2/5,
    # wrong majority) -> high-confidence queries are the correct ones -> AURC beats random.
    s = analyze(_matrix([{"a": [1, 1, 1, 1, 1]}, {"a": [1, 1, 1, 1, 1]},
                         {"a": [1, 1, 0, 0, 0]}, {"a": [1, 1, 0, 0, 0]}]))
    p = s.per_model[0]
    assert p.base_accuracy == pytest.approx(0.5)          # 2 of 4 majority-correct
    assert p.aurc < p.random_aurc and p.aurc_gain > 0     # confidence ranks the right ones first
    assert p.accuracy_at_coverage[0.5] == pytest.approx(1.0)   # top-half all correct
    assert p.abstention_gain > 0
    assert s.any_confidence_informative is True


def test_confidence_uninformative_when_agreement_anticorrelates():
    # high agreement on WRONG queries (0/5 -> confident but wrong), split on right ones:
    # ranking by confidence puts wrong answers first -> AURC no better than random.
    s = analyze(_matrix([{"a": [0, 0, 0, 0, 0]}, {"a": [0, 0, 0, 0, 0]},
                         {"a": [1, 1, 1, 0, 0]}, {"a": [1, 1, 1, 0, 0]}]))
    p = s.per_model[0]
    assert p.base_accuracy == pytest.approx(0.5)
    assert p.aurc_gain <= 0                                # confidence is anti-informative
    assert s.any_confidence_informative is False


def test_k1_confidence_is_degenerate_but_safe():
    # K=1: confidence is always 1.0 (no self-consistency signal) -> AURC == random_aurc.
    s = analyze(_matrix([{"a": [1]}, {"a": [0]}, {"a": [1]}, {"a": [0]}]))
    p = s.per_model[0]
    assert p.base_accuracy == pytest.approx(0.5)
    assert p.aurc == pytest.approx(p.random_aurc) and p.aurc_gain == pytest.approx(0.0)


def test_multi_model_picks_best_aurc():
    # a: informative; b: all correct (AURC 0). best AURC -> b.
    s = analyze(_matrix([{"a": [1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1]},
                         {"a": [1, 1, 0, 0, 0], "b": [1, 1, 1, 1, 1]}]))
    assert s.best_aurc_model == "b" and s.best_aurc == pytest.approx(0.0)
    assert set(s.models) == {"a", "b"}


def test_empty_matrix():
    s = analyze({"benchmark": "x", "tasks": []})
    assert s.n_questions == 0 and s.best_aurc_model is None
    assert s.any_confidence_informative is False


# --------------------------------------------------------------------------- #
# real committed data: runs and produces a sane curve
# --------------------------------------------------------------------------- #
def test_real_math500_matrix_if_present():
    p = _REPO / "experiments" / "final" / "oracle_matrix_math500.json"
    if not p.exists():
        pytest.skip("real oracle_matrix_math500.json not present")
    s = analyze(json.loads(p.read_text()))
    assert s.n_questions == 120 and s.k == 5 and len(s.models) == 3
    for pm in s.per_model:
        # coverage 1.0 accuracy must equal the base majority@K accuracy.
        assert pm.accuracy_at_coverage[1.0] == pytest.approx(pm.base_accuracy)
        assert 0.0 <= pm.aurc <= 1.0


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_report():
    s = analyze(_matrix([{"a": [1, 1, 1, 1, 1]}, {"a": [1, 1, 0, 0, 0]}]))
    md = render(s)
    assert "selective prediction" in md.lower() and "AURC" in md
    assert "acc@0.5" in md and "risk-coverage" in md.lower()
    assert render(analyze({"benchmark": "x", "tasks": []})).strip().endswith("(no matrix data)_")
