"""Offline coverage for the SepCMAES wrapper's guards and ask/tell/best/run paths.

`test_sep_cmaes_seed.py` pins the reproducible-sampling contract, but the input
guards (dimension / x0-shape / tell-length), the introspection accessors, the
``best()`` before-any-tell error, and the standalone ``run`` driver were uncovered
(sep_cmaes.py at 77%). These are the separable CMA-ES optimizer that searches the
13,312-dim θ — a bad guard or a mis-reported best would corrupt training silently.

Pure/numpy + pycma only (no torch, no GPU, no network); a tiny synthetic sphere
objective stands in for the real SLM+pool fitness, exactly as smoke test S7 does.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from trinity.optim.sep_cmaes import SepCMAES, default_popsize, run


def test_no_torch_imported():
    assert "torch" not in sys.modules, "sep-CMA-ES wrapper must import without torch"


def _neg_sphere(x: np.ndarray) -> float:
    """Negative squared norm — maximized (at 0) by the origin."""
    return -float(np.dot(x, x))


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_default_popsize_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n must be >= 1"):
        default_popsize(0)


def test_default_popsize_spec_value():
    assert default_popsize(13312) == 33  # ceil(4 + 3 ln n)


def test_init_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="n must be >= 1"):
        SepCMAES(n=0)


def test_init_rejects_seed_out_of_range():
    with pytest.raises(ValueError, match="seed must be in"):
        SepCMAES(n=4, seed=-1)


def test_init_rejects_wrong_x0_shape():
    with pytest.raises(ValueError, match=r"x0 must have shape"):
        SepCMAES(n=4, x0=np.zeros(5))


def test_tell_rejects_length_mismatch():
    opt = SepCMAES(n=4, maxiter=3, seed=0)
    sols = opt.ask()
    with pytest.raises(ValueError, match="equal length"):
        opt.tell(sols, [0.1])  # one fitness for many solutions


def test_best_before_any_tell_raises():
    opt = SepCMAES(n=4, maxiter=3, seed=0)
    with pytest.raises(RuntimeError, match="before any tell"):
        opt.best()


# --------------------------------------------------------------------------- #
# introspection + a manual ask/tell/best cycle
# --------------------------------------------------------------------------- #
def test_popsize_and_iteration_accessors():
    opt = SepCMAES(n=4, popsize=6, maxiter=3, seed=0)
    assert opt.popsize == 6
    assert opt.iteration == 0
    assert opt.stop() is False


def test_ask_tell_updates_iteration_and_best():
    opt = SepCMAES(n=4, popsize=6, maxiter=5, seed=1)
    sols = opt.ask()
    assert len(sols) == 6
    assert all(s.shape == (4,) for s in sols)
    opt.tell(sols, [_neg_sphere(x) for x in sols])
    assert opt.iteration == 1
    best_x, best_f = opt.best()
    assert best_x.shape == (4,)
    assert best_f <= 0.0                      # sphere is maximized at 0
    # x0 defaults to zeros when not supplied.
    opt0 = SepCMAES(n=4, maxiter=1, seed=0)
    assert opt0.popsize == default_popsize(4)


# --------------------------------------------------------------------------- #
# run() driver
# --------------------------------------------------------------------------- #
def test_run_reports_history_and_maximizes(capsys):
    best_x, best_f, history = run(_neg_sphere, n=4, maxiter=3, seed=0, verbose=True)
    assert best_x.shape == (4,)
    assert best_f <= 0.0
    assert len(history) >= 1
    assert set(history[0]) == {
        "iteration", "best_fitness", "gen_best_fitness", "gen_mean_fitness"
    }
    # best-so-far is monotone non-decreasing.
    curve = [h["best_fitness"] for h in history]
    assert curve == sorted(curve)
    # verbose=True prints a per-generation line.
    assert "sep-CMA-ES" in capsys.readouterr().out


def test_run_accepts_explicit_x0_and_popsize():
    best_x, _bf, history = run(
        _neg_sphere, n=4, x0=np.full(4, 0.1), popsize=6, maxiter=2, seed=0
    )
    assert best_x.shape == (4,)
    assert len(history) >= 1
