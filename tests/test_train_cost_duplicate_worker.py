"""Regression: a pool that lists a model more than once must not drop its calls (issue #270).

`estimate_cmaes_cost` keyed `per_model_usd` by model name (collapsing duplicates)
while dividing the calls by the slot count, so a repeated worker's share of the
`worker_calls` was silently un-priced and `total_usd` under-counted the run.
Pure Python — no network, no GPU, no torch.
"""
from __future__ import annotations

from trinity.train_cost import estimate_cmaes_cost

PRICES = {"cheap": (0.1, 0.2), "pricey": (3.0, 6.0)}


def _est(worker_names, **kw):
    return estimate_cmaes_cost(
        population_size=10, m_cma=1, generations=1,
        worker_names=worker_names, prices=PRICES,
        avg_turns=3, avg_prompt_tokens=1_000_000, avg_completion_tokens=1_000_000,
        **kw,
    )


def test_duplicate_worker_keeps_all_calls_priced():
    # 3 slots, 30 worker calls -> 10 per slot. per-call: cheap=0.3, pricey=9.0.
    # cheap listed twice -> 20 calls * 0.3 = 6.0 ; pricey 10 * 9.0 = 90.0 ; total 96.0.
    e = _est(["cheap", "cheap", "pricey"])
    assert e.worker_calls == 30
    assert e.per_model_usd["cheap"] == 6.0
    assert e.per_model_usd["pricey"] == 90.0
    assert e.total_usd == 96.0


def test_two_of_same_worker_prices_both_slots():
    # ['a', 'a'] must price BOTH slots, not one (previously reported half).
    e = _est(["cheap", "cheap"])
    # 30 calls / 2 slots = 15 each -> cheap gets all 30 * 0.3 = 9.0.
    assert e.total_usd == 9.0
    assert e.per_model_usd["cheap"] == 9.0


def test_total_equals_slot_blended_price_times_calls():
    # total_usd must equal worker_calls * (mean per-call cost over the SLOTS).
    names = ["cheap", "cheap", "pricey"]
    e = _est(names)
    per_call = {"cheap": 0.3, "pricey": 9.0}
    slot_blend = sum(per_call[n] for n in names) / len(names)
    assert abs(e.total_usd - round(e.worker_calls * slot_blend, 2)) < 1e-6


def test_distinct_pool_is_unchanged():
    # Regression guard: a distinct pool must price exactly as before.
    e = _est(["cheap", "pricey"])
    # 30 / 2 = 15 each -> cheap 15*0.3=4.5, pricey 15*9.0=135.0, total 139.5.
    assert e.per_model_usd == {"cheap": 4.5, "pricey": 135.0}
    assert e.total_usd == 139.5
