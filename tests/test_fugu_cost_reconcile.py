"""Offline tests for the fugu projected-vs-actual cost reconciliation. No network, no GPU."""
from __future__ import annotations

import json

import pytest

from trinity.fugu.cost import estimate_eval_cost
from trinity.fugu.cost_reconcile import reconcile_projection, render

_WORKERS = ["deepseek-v4-flash", "qwen3.5-35b-a3b"]


def _projection():
    # rollouts = reps*n_tasks = 10; worker_calls = round(10*2.5)=25; conductor_calls=10.
    return estimate_eval_cost(worker_names=_WORKERS, n_tasks=10, reps=1)


def _actual(**over):
    a = {"spend_usd": 0.0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    a.update(over)
    return a


def test_projected_calls_are_worker_plus_conductor():
    est = _projection()
    rec = reconcile_projection(est, _actual(llm_calls=35))
    calls = rec.field("llm_calls")
    assert calls is not None
    assert calls.projected == est["worker_calls"] + est["conductor_calls"]  # 25 + 10 = 35
    assert calls.actual == 35 and not calls.overrun and not calls.underrun


def test_spend_overrun_is_flagged_with_ratio():
    est = _projection()
    rec = reconcile_projection(est, _actual(spend_usd=2 * est["total_usd"], llm_calls=35))
    spend = rec.field("spend_usd")
    assert spend is not None and spend.overrun and not spend.underrun
    assert spend.ratio == pytest.approx(2.0)
    assert spend.delta == pytest.approx(est["total_usd"])
    assert not rec.within_tolerance


def test_spend_underrun_is_flagged():
    est = _projection()
    rec = reconcile_projection(est, _actual(spend_usd=0.1 * est["total_usd"], llm_calls=35))
    spend = rec.field("spend_usd")
    assert spend is not None and spend.underrun and not spend.overrun
    assert not rec.within_tolerance


def test_token_fields_are_reconstructed_from_assumptions():
    est = _projection()
    # projected prompt = 25*1200 + 10*700 = 37000; completion = 25*800 + 10*250 = 22500.
    rec = reconcile_projection(est, _actual(
        spend_usd=est["total_usd"], llm_calls=35, prompt_tokens=37000, completion_tokens=22500))
    pt = rec.field("prompt_tokens")
    ct = rec.field("completion_tokens")
    assert pt is not None and pt.projected == pytest.approx(37000)
    assert ct is not None and ct.projected == pytest.approx(22500)
    assert rec.within_tolerance                       # everything matches


def test_within_tolerance_when_everything_matches():
    est = _projection()
    rec = reconcile_projection(est, _actual(
        spend_usd=est["total_usd"], llm_calls=est["worker_calls"] + est["conductor_calls"],
        prompt_tokens=37000, completion_tokens=22500))
    assert rec.within_tolerance
    assert all(not f.overrun and not f.underrun for f in rec.fields)


def test_token_fields_absent_without_assumptions():
    # A bare projection with no assumptions -> only spend + calls are compared.
    rec = reconcile_projection(
        {"total_usd": 1.0, "worker_calls": 10, "conductor_calls": 5}, _actual(llm_calls=15))
    assert {f.field for f in rec.fields} == {"spend_usd", "llm_calls"}


def test_zero_projection_gives_none_ratio_and_does_not_crash():
    rec = reconcile_projection({"total_usd": 0.0, "worker_calls": 0, "conductor_calls": 0},
                               _actual(spend_usd=5.0, llm_calls=3))
    spend = rec.field("spend_usd")
    assert spend is not None and spend.ratio is None
    assert spend.overrun                              # 5.0 > 0*(1+tol)=0


def test_missing_actual_fields_default_to_zero():
    est = _projection()
    rec = reconcile_projection(est, {})               # empty actual, must not raise
    spend = rec.field("spend_usd")
    assert spend is not None and spend.actual == 0.0 and spend.underrun


def test_render_and_to_dict():
    est = _projection()
    rec = reconcile_projection(est, _actual(spend_usd=2 * est["total_usd"], llm_calls=35,
                                            prompt_tokens=37000, completion_tokens=22500))
    md = render(rec)
    assert "cost reconciliation" in md.lower() and "OUT of tolerance" in md
    assert "spend_usd" in md and "2.00x" in md
    d = rec.to_dict()
    assert json.loads(json.dumps(d))["within_tolerance"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
