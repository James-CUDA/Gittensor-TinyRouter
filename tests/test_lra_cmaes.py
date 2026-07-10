"""Offline tests for LRA-CMA-ES learning-rate adaptation (IMPROVEMENTS.md #4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from trinity.optim.lra import (
    LRAConfig,
    LRAState,
    adapt_eta,
    apply_learning_rate,
    capture_baseline_opts,
    estimate_snr,
)
from trinity.optim.sep_cmaes import SepCMAES, run

cma = pytest.importorskip("cma", reason="pycma is required for LRA-CMA-ES tests")

_N_SMALL = 8
_POP = 6
_GENS = 12


def _enabled_lra(**overrides) -> LRAConfig:
    base = dict(
        enabled=True,
        target_snr=1.0,
        beta=0.3,
        eta_min=0.05,
        eta_max=1.0,
        initial_eta=1.0,
    )
    base.update(overrides)
    return LRAConfig(**base)


def _sphere(theta_star: np.ndarray):
    def objective(x: np.ndarray) -> float:
        d = x - theta_star
        return -float(np.dot(d, d))

    return objective


def _noisy_sphere(theta_star: np.ndarray, rng: np.random.Generator, noise: float):
    def objective(x: np.ndarray) -> float:
        d = x - theta_star
        base = -float(np.dot(d, d))
        return base + float(rng.normal(0.0, noise))

    return objective


def _flat_binary(rng: np.random.Generator, p: float = 0.35):
    def objective(_x: np.ndarray) -> float:
        return 1.0 if rng.random() < p else 0.0

    return objective


# --------------------------------------------------------------------------- #
# Pure LRA math
# --------------------------------------------------------------------------- #


def test_lra_config_from_dict_defaults_disabled():
    cfg = LRAConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.target_snr == 1.0


def test_lra_config_from_dict_parses_enabled_block():
    cfg = LRAConfig.from_dict({"enabled": True, "eta_min": 0.1, "beta": 0.4})
    assert cfg.enabled is True
    assert cfg.eta_min == 0.1
    assert cfg.beta == 0.4


def test_estimate_snr_zero_when_no_signal():
    assert estimate_snr(0.0, 0.0) == 0.0


def test_estimate_snr_grows_with_consistent_shift():
    low = estimate_snr(0.01, 0.5)
    high = estimate_snr(0.2, 0.01)
    assert high > low


def test_adapt_eta_clamps_to_bounds():
    cfg = LRAConfig(enabled=True, target_snr=1.0, eta_min=0.1, eta_max=0.5)
    assert adapt_eta(0.0, cfg, 0.2) == pytest.approx(0.1)
    assert adapt_eta(100.0, cfg, 0.2) == pytest.approx(0.5)


def test_capture_baseline_opts_reads_pycma_keys():
    es = cma.CMAEvolutionStrategy(
        [0.0] * _N_SMALL,
        0.5,
        {"CMA_diagonal": True, "popsize": _POP, "seed": np.nan, "verbose": -9},
    )
    baseline = capture_baseline_opts(es)
    assert "CMA_cmean" in baseline
    assert baseline["CMA_cmean"] > 0


def test_apply_learning_rate_scales_opts():
    es = cma.CMAEvolutionStrategy(
        [0.0] * _N_SMALL,
        0.5,
        {"CMA_diagonal": True, "popsize": _POP, "seed": np.nan, "verbose": -9},
    )
    baseline = capture_baseline_opts(es)
    apply_learning_rate(es, baseline, 0.25)
    for key, base in baseline.items():
        assert es.opts[key] == pytest.approx(base * 0.25)


def test_lra_state_first_observation_keeps_initial_eta():
    es = cma.CMAEvolutionStrategy(
        [0.0] * _N_SMALL,
        0.5,
        {"CMA_diagonal": True, "popsize": _POP, "seed": np.nan, "verbose": -9},
    )
    cfg = _enabled_lra(initial_eta=0.8)
    state = LRAState()
    state.reset(cfg)
    baseline = capture_baseline_opts(es)
    diag = state.observe(0.2, cfg, es, baseline, iteration=1)
    assert diag["eta"] == pytest.approx(0.8)
    assert diag["snr"] == 0.0


def test_lra_state_shrinks_eta_on_flat_noise():
    es = cma.CMAEvolutionStrategy(
        [0.0] * _N_SMALL,
        0.5,
        {"CMA_diagonal": True, "popsize": _POP, "seed": np.nan, "verbose": -9},
    )
    cfg = _enabled_lra(beta=0.5, target_snr=2.0)
    state = LRAState()
    state.reset(cfg)
    baseline = capture_baseline_opts(es)
    state.observe(0.3, cfg, es, baseline, iteration=1)
    for i in range(8):
        state.observe(0.3, cfg, es, baseline, iteration=i + 2)
    assert state.eta < cfg.initial_eta


# --------------------------------------------------------------------------- #
# SepCMAES integration
# --------------------------------------------------------------------------- #


def test_lra_disabled_matches_legacy_first_population():
  legacy = np.asarray(
      SepCMAES(n=_N_SMALL, sigma0=0.2, seed=7, popsize=_POP, lra=None).ask()
  )
  disabled = np.asarray(
      SepCMAES(
          n=_N_SMALL,
          sigma0=0.2,
          seed=7,
          popsize=_POP,
          lra=LRAConfig(enabled=False),
      ).ask()
  )
  assert np.allclose(legacy, disabled)


def test_lra_disabled_preserves_run_history():
    star = np.random.default_rng(0).standard_normal(_N_SMALL) * 0.1
    obj = _sphere(star)
    _, f_off, hist_off = run(
        obj, _N_SMALL, seed=3, popsize=_POP, maxiter=_GENS, lra=None
    )
    _, f_disabled, hist_disabled = run(
        obj,
        _N_SMALL,
        seed=3,
        popsize=_POP,
        maxiter=_GENS,
        lra=LRAConfig(enabled=False),
    )
    assert f_off == pytest.approx(f_disabled)
    assert len(hist_off) == len(hist_disabled)
    for a, b in zip(hist_off, hist_disabled):
        assert a["best_fitness"] == pytest.approx(b["best_fitness"])


def test_lra_enabled_records_diagnostics_in_history():
    star = np.random.default_rng(1).standard_normal(_N_SMALL) * 0.05
    _, _, hist = run(
        _sphere(star),
        _N_SMALL,
        seed=11,
        popsize=_POP,
        maxiter=6,
        lra=_enabled_lra(),
    )
    assert len(hist) == 6
    assert all("lra" in row for row in hist)
    assert all("eta" in row["lra"] for row in hist)


def test_lra_enabled_is_deterministic_for_fixed_seed():
    star = np.random.default_rng(2).standard_normal(_N_SMALL) * 0.05
    cfg = _enabled_lra()
    bx1, bf1, h1 = run(
        _sphere(star), _N_SMALL, seed=42, popsize=_POP, maxiter=8, lra=cfg
    )
    bx2, bf2, h2 = run(
        _sphere(star), _N_SMALL, seed=42, popsize=_POP, maxiter=8, lra=cfg
    )
    assert np.allclose(bx1, bx2)
    assert bf1 == pytest.approx(bf2)
    assert h1[-1]["lra"]["eta"] == pytest.approx(h2[-1]["lra"]["eta"])


def test_lra_improves_noisy_sphere_median_fitness():
    """On a noisy objective LRA should not underperform a fixed-rate baseline."""
    star = np.random.default_rng(5).standard_normal(_N_SMALL) * 0.08
    cfg = _enabled_lra(beta=0.4, target_snr=0.8, eta_min=0.1)
    lra_scores: list[float] = []
    base_scores: list[float] = []
    for seed in range(5):
        rng = np.random.default_rng(seed + 100)
        obj = _noisy_sphere(star, rng, noise=0.15)
        _, bf_lra, _ = run(
            obj, _N_SMALL, seed=seed, popsize=_POP, maxiter=_GENS, lra=cfg
        )
        _, bf_base, _ = run(
            obj, _N_SMALL, seed=seed, popsize=_POP, maxiter=_GENS, lra=None
        )
        lra_scores.append(bf_lra)
        base_scores.append(bf_base)
    assert np.median(lra_scores) >= np.median(base_scores)


def test_lra_maintains_progress_under_flat_binary_noise():
    """Flat binary rewards should not collapse best_f to zero instantly with LRA."""
    cfg = _enabled_lra(beta=0.5, eta_min=0.05, target_snr=1.5)
    _, best_f, hist = run(
        _flat_binary(np.random.default_rng(9), p=0.3),
        _N_SMALL,
        seed=4,
        popsize=_POP,
        maxiter=_GENS,
        lra=cfg,
    )
    assert best_f >= 0.0
    assert any(row["lra"]["eta"] < 1.0 for row in hist[2:])
    assert hist[-1]["best_fitness"] >= hist[0]["best_fitness"]


def test_manual_tell_loop_exposes_lra_history():
    opt = SepCMAES(
        n=_N_SMALL,
        sigma0=0.3,
        seed=0,
        popsize=_POP,
        lra=_enabled_lra(),
    )
    for gen in range(5):
        sols = opt.ask()
        fits = [0.1 * gen + 0.01 * i for i in range(len(sols))]
        opt.tell(sols, fits)
    assert len(opt.lra_history) == 5
    assert opt.lra_history[-1]["gen_mean_fitness"] == pytest.approx(
        float(np.mean(fits))
    )


def test_lra_eta_never_exits_configured_bounds_across_run():
    cfg = _enabled_lra(eta_min=0.2, eta_max=0.9, beta=0.35)
    _, _, hist = run(
        _flat_binary(np.random.default_rng(12), p=0.4),
        _N_SMALL,
        seed=8,
        popsize=_POP,
        maxiter=15,
        lra=cfg,
    )
    etas = [row["lra"]["eta"] for row in hist]
    assert all(cfg.eta_min - 1e-9 <= eta <= cfg.eta_max + 1e-9 for eta in etas)


def test_lra_snr_recovers_after_step_change():
    state = LRAState()
    cfg = _enabled_lra(beta=0.6, target_snr=1.0)
    state.reset(cfg)
    es = cma.CMAEvolutionStrategy(
        [0.0] * 4,
        0.5,
        {"CMA_diagonal": True, "popsize": 4, "seed": np.nan, "verbose": -9},
    )
    baseline = capture_baseline_opts(es)
    state.observe(0.1, cfg, es, baseline, iteration=1)
    for _ in range(6):
        state.observe(0.1, cfg, es, baseline, iteration=2)
    flat_eta = state.eta
    for i in range(4):
        state.observe(0.1 + 0.05 * (i + 1), cfg, es, baseline, iteration=10 + i)
    assert state.eta >= flat_eta
    assert estimate_snr(state.s1, state.s2) > 0.0


def test_import_surface_exports_lra_config():
    from trinity.optim import LRAConfig as Exported

    assert Exported.from_dict({"enabled": True}).enabled is True
