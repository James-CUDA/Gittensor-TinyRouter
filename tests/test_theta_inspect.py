"""Offline tests for the theta diagnostics. No network, no GPU (numpy only)."""
from __future__ import annotations

import numpy as np
import pytest

from trinity.coordinator import params as P
from trinity.theta_inspect import inspect_theta


def _spec():
    # Small spec so the tests are fast and explicit: head (2x3)=6 + svf 4 = 10.
    return P.make_spec(n_a=2, d_h=3, n_svf=4)


def _theta(head, svf):
    return P.pack(np.asarray(head, dtype=float), np.asarray(svf, dtype=float))


# ---------------------------------------------------------------------------
# init detection (the whole point)
# ---------------------------------------------------------------------------
def test_initial_theta_is_flagged_as_untrained():
    spec = _spec()
    report = inspect_theta(P.initial_theta(spec), spec)
    assert not report.trained
    assert not report.head_trained and not report.svf_trained
    assert report.head.at_init and report.svf.at_init
    assert "uniform policy" in " ".join(report.warnings)
    assert "not adapted" in " ".join(report.warnings)


def test_a_trained_theta_is_flagged_as_trained():
    spec = _spec()
    head = np.full((2, 3), 0.3)
    svf = np.array([1.2, 0.8, 1.1, 0.9])
    report = inspect_theta(_theta(head, svf), spec)
    assert report.trained and report.head_trained and report.svf_trained
    assert report.warnings == []
    assert report.svf.n_moved == 4 and report.head.n_moved == 6


def test_head_trained_but_svf_untouched():
    spec = _spec()
    report = inspect_theta(_theta(np.full((2, 3), 0.5), np.ones(4)), spec)
    assert report.head_trained and not report.svf_trained
    assert report.trained  # either block moving counts as trained
    assert any("SVF scales are still at their 1.0 init" in w for w in report.warnings)


def test_svf_trained_but_head_zero():
    spec = _spec()
    report = inspect_theta(_theta(np.zeros((2, 3)), np.array([1.5, 1, 1, 1])), spec)
    assert report.svf_trained and not report.head_trained
    assert report.svf.n_moved == 1
    assert any("uniform policy" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# block stats
# ---------------------------------------------------------------------------
def test_block_stats_norms_and_distance():
    spec = _spec()
    report = inspect_theta(_theta(np.zeros((2, 3)), np.ones(4)), spec)
    # head at 0: norm 0, dist_from_init 0.
    assert report.head.l2_norm == 0.0 and report.head.dist_from_init == 0.0
    # svf at 1: norm 2 (sqrt(4)), but dist_from_init 0 (init is 1).
    assert report.svf.l2_norm == pytest.approx(2.0)
    assert report.svf.dist_from_init == pytest.approx(0.0)
    assert report.svf.frac_moved == 0.0


def test_frac_moved_is_partial():
    spec = _spec()
    report = inspect_theta(_theta(np.zeros((2, 3)), np.array([1.5, 1.0, 1.0, 1.0])), spec)
    assert report.svf.n_moved == 1 and report.svf.frac_moved == 0.25


# ---------------------------------------------------------------------------
# non-finite detection
# ---------------------------------------------------------------------------
def test_nonfinite_entries_are_reported():
    spec = _spec()
    head = np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]])
    report = inspect_theta(_theta(head, np.ones(4)), spec)
    assert not report.finite
    assert report.n_nonfinite == 2
    assert any("non-finite" in w for w in report.warnings)
    # A NaN entry never counts as "at init".
    assert report.head.n_at_init == 4  # the four real zeros


# ---------------------------------------------------------------------------
# shape validation (maintainer lesson: reject wrong shapes explicitly)
# ---------------------------------------------------------------------------
def test_wrong_length_raises():
    spec = _spec()
    with pytest.raises(ValueError, match="entries but spec.n_total"):
        inspect_theta(np.zeros(spec.n_total + 1), spec)


def test_non_1d_theta_raises():
    spec = _spec()
    with pytest.raises(ValueError, match="1-D"):
        inspect_theta(np.zeros((2, 5)), spec)


def test_default_spec_is_the_canonical_13312():
    # No explicit spec -> params.make_spec() (head 6144 + svf 7168 = 13312).
    report = inspect_theta(P.initial_theta(P.make_spec()))
    assert report.n_total == 13312
    assert report.head.size == 6144 and report.svf.size == 7168


def test_report_roundtrips_to_dict():
    spec = _spec()
    d = inspect_theta(P.initial_theta(spec), spec).to_dict()
    assert d["trained"] is False and d["finite"] is True
    assert d["head"]["at_init"] and d["svf"]["at_init"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
