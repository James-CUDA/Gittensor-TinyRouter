"""Regression: union_oracle must reject a RAGGED pool, not a reordered same set.

The guard is documented to require the benchmarks to share one model *set*, but it
compared the model *lists* order-sensitively, so two benchmarks listing the same
models in a different per_model insertion order were wrongly rejected -- even
though the equal-weight average looks each accuracy up by name. No network/GPU.
"""
from __future__ import annotations

import pytest

from trinity.analysis.union_oracle import union_oracle


def _matrix(benchmark, per_query):
    # Preserve the given key order so per_model insertion order (and thus
    # oracle.models order) reflects the test's intent.
    tasks = [{"id": f"q{i}", "per_model": {m: [v] for m, v in pm.items()}}
             for i, pm in enumerate(per_query)]
    return {"benchmark": benchmark, "tasks": tasks}


def test_same_model_set_reordered_is_accepted():
    a = _matrix("math", [{"x": 1, "y": 0}])
    b = _matrix("mmlu", [{"y": 1, "x": 0}])            # same set {x,y}, keys reordered
    s = union_oracle([a, b])                            # previously raised ValueError
    assert set(s.models) == {"x", "y"}
    assert s.equal_weight_per_model_accuracy["x"] == pytest.approx(0.5)
    assert s.equal_weight_per_model_accuracy["y"] == pytest.approx(0.5)


def test_reordered_pool_matches_the_ordered_pool_exactly():
    ordered = union_oracle([_matrix("math", [{"x": 1, "y": 0}]),
                            _matrix("mmlu", [{"x": 0, "y": 1}])])
    reordered = union_oracle([_matrix("math", [{"x": 1, "y": 0}]),
                              _matrix("mmlu", [{"y": 1, "x": 0}])])
    assert reordered.equal_weight_per_model_accuracy == ordered.equal_weight_per_model_accuracy
    assert reordered.best_single == ordered.best_single
    assert reordered.routing_oracle == ordered.routing_oracle


def test_genuinely_ragged_pool_is_still_rejected():
    a = _matrix("math", [{"x": 1, "y": 0}])
    b = _matrix("mmlu", [{"x": 1, "z": 0}])            # different set {x,z}
    with pytest.raises(ValueError, match="models"):
        union_oracle([a, b])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
