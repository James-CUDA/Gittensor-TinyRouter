"""``abstention_gain`` reports the documented mild-abstention level (acc@0.8).

``ModelSelective.abstention_gain`` is documented as ``acc@0.8 - acc@1.0`` — the
lift from dropping the least-confident 20%. With the default coverages
``(1.0, 0.8, 0.5)`` the metric must therefore use the *highest* partial coverage
(``0.8``), i.e. the mildest abstention. These tests pin it to that contract and
guard against selecting the deepest coverage (``0.5``) instead.

Pure / offline — numpy only, no torch, no network.
"""
from __future__ import annotations

import pytest

from trinity.analysis.selective import analyze


def _matrix(per_model_by_q, benchmark="math500"):
    """per_model_by_q: list of {model: [0/1,...K]} dicts (one per question)."""
    return {"benchmark": benchmark,
            "tasks": [{"id": f"q{i}", "per_model": pm} for i, pm in enumerate(per_model_by_q)]}


# Top 5 queries: fully-confident and correct (conf 1.0). Bottom 5: conf 0.6, of
# which 2 are majority-correct and 3 wrong. This makes acc@0.5 (top half) differ
# from acc@0.8 (top 80%), so the choice of partial coverage is observable:
#   acc@1.0 = 0.700, acc@0.8 = 0.775, acc@0.5 = 1.000
_SPLIT_MATRIX = _matrix(
    [{"a": [1, 1, 1, 1, 1]}] * 5      # 5/5 solves -> conf 1.0, correct
    + [{"a": [1, 1, 1, 0, 0]}] * 2    # 3/5 solves -> conf 0.6, correct
    + [{"a": [1, 1, 0, 0, 0]}] * 3    # 2/5 solves -> conf 0.6, wrong
)


def test_abstention_gain_uses_the_highest_partial_coverage():
    p = analyze(_SPLIT_MATRIX).per_model[0]
    # The two partial levels genuinely differ here, so the metric is unambiguous.
    assert p.accuracy_at_coverage[0.8] != pytest.approx(p.accuracy_at_coverage[0.5])
    # Documented contract: gain is the MILDEST abstention == acc@0.8 - acc@1.0.
    assert p.abstention_gain == pytest.approx(
        p.accuracy_at_coverage[0.8] - p.accuracy_at_coverage[1.0])
    # ...and NOT the deepest coverage (acc@0.5), which the old min() wrongly used.
    assert p.abstention_gain != pytest.approx(
        p.accuracy_at_coverage[0.5] - p.accuracy_at_coverage[1.0])


def test_abstention_gain_exact_values():
    p = analyze(_SPLIT_MATRIX).per_model[0]
    assert p.accuracy_at_coverage[1.0] == pytest.approx(0.700)
    assert p.accuracy_at_coverage[0.8] == pytest.approx(0.775)
    assert p.accuracy_at_coverage[0.5] == pytest.approx(1.000)
    assert p.abstention_gain == pytest.approx(0.075)


def test_single_partial_coverage_is_unambiguous():
    # Only one partial level -> the highest and deepest partial coincide.
    p = analyze(_SPLIT_MATRIX, coverages=(1.0, 0.8)).per_model[0]
    assert p.abstention_gain == pytest.approx(
        p.accuracy_at_coverage[0.8] - p.accuracy_at_coverage[1.0])


def test_full_coverage_only_gives_zero_gain():
    # No partial coverage configured -> nothing is dropped, so the gain is 0.
    p = analyze(_SPLIT_MATRIX, coverages=(1.0,)).per_model[0]
    assert p.abstention_gain == pytest.approx(0.0)
