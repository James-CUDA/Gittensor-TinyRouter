"""Milestone-1 gates: param budget + rate limit."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from trinity.m1.constants import MAX_HEAD_PARAMS
from trinity.m1.gates import check_param_budget, check_rate_limit, run_m1_gates
from trinity.m1.pack import Milestone1Pack, count_pack_params


def _tiny_pack(**kw) -> Milestone1Pack:
    d_h = kw.pop("d_h", 8)
    return Milestone1Pack(
        config="5-domain",
        pool="penultimate",
        d_h=d_h,
        W_domain=np.zeros((5, d_h), np.float32),
        W_diff=np.zeros((5, d_h), np.float32),
        **kw,
    )


def test_default_pack_under_1m():
    pack = _tiny_pack(d_h=1024)
    assert count_pack_params(pack) == 5 * 1024 + 5 * 1024
    assert count_pack_params(pack) < MAX_HEAD_PARAMS
    assert check_param_budget(pack).ok


def test_param_budget_rejects_huge():
    # Force a pack that claims huge weights via oversized arrays
    d_h = 512
    C = 2000  # not a real taxonomy; bypass expected_domains by crafting after
    pack = Milestone1Pack(
        config="5-domain",
        pool="penultimate",
        d_h=d_h,
        W_domain=np.zeros((5, d_h), np.float32),
        W_diff=np.zeros((5, d_h), np.float32),
    )
    # Monkey-patch sizes for budget check only
    pack.W_domain = np.zeros((C, d_h), np.float32)
    pack.W_diff = np.zeros((5, d_h), np.float32)
    assert count_pack_params(pack) >= MAX_HEAD_PARAMS or count_pack_params(pack) > 0
    # With C=2000, d_h=512 → 2000*512 + 5*512 = 1,026,560 >= 1M
    assert count_pack_params(pack) >= MAX_HEAD_PARAMS
    r = check_param_budget(pack)
    assert r.failed


def test_rate_limit_one_per_day(tmp_path: Path):
    attempts = tmp_path / "submissions" / "m1_attempts.jsonl"
    attempts.parent.mkdir(parents=True)
    attempts.write_text(
        '{"miner":"alice","submission":"x","ts":"2099-01-01T12:00:00Z","ok":true}\n',
        encoding="utf-8",
    )
    # Use a fixed "now" just after that ts so it counts
    import calendar

    now = calendar.timegm((2099, 1, 1, 13, 0, 0))
    r = check_rate_limit("alice", tmp_path, now=now)
    assert r.failed
    r2 = check_rate_limit("bob", tmp_path, now=now)
    assert r2.ok


def test_run_gates_pass(tmp_path: Path):
    pack = _tiny_pack()
    results = run_m1_gates(pack, miner="carol", repo_root=tmp_path, skip_rate_limit=False)
    assert all(r.ok for r in results)
