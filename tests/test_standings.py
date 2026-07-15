"""Offline tests for the cross-benchmark competition standings.

Synthetic leaderboards (the leaderboard.json schema); the headline case is that a
generalist strong on every benchmark outranks a specialist who tops one. stdlib only, no
torch/network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.standings import compute_standings, load_standings, render

_REPO = Path(__file__).resolve().parents[1]


def _lb(benchmarks: dict) -> dict:
    return {"updated_at": "2026-07-13T00:00:00Z", "benchmarks": benchmarks}


def _win(miner, score, pr, merged=True):
    return {"miner": miner, "generation": 1, "score": score, "pr": pr,
            "merged": merged, "timestamp": "2026-07-13T00:00:00Z"}


def test_module_imports_without_torch():
    # A global sys.modules check is unreliable where torch IS installed (CI): another
    # test imports it into the shared process before this one runs. Verify in a CLEAN
    # subprocess that importing this module alone never pulls in torch.
    code = ("import sys; sys.path.insert(0, 'src'); import trinity.standings; "
            "assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# the equal-weight thesis: generalist beats specialist
# --------------------------------------------------------------------------- #
def test_generalist_outranks_specialist():
    lb = _lb({
        "math500": {"best_miner": "bob", "history": [_win("alice", 0.80, 1), _win("bob", 0.85, 2)]},
        "mmlu": {"best_miner": "alice", "history": [_win("alice", 0.90, 3), _win("carol", 0.70, 4)]},
    })
    s = compute_standings(lb)
    assert [m.miner for m in s.miners] == ["alice", "bob", "carol"]
    assert s.leader == "alice"
    assert s.miners[0].overall == pytest.approx(0.85)    # (0.80 + 0.90) / 2
    assert s.miners[1].overall == pytest.approx(0.425)   # (0.85 + 0) / 2 — specialist penalised
    assert s.miners[0].benchmarks_led == 1 and s.miners[1].benchmarks_led == 1


def test_only_merged_wins_count():
    lb = _lb({"math500": {"history": [_win("bob", 0.99, 1, merged=False), _win("bob", 0.60, 2)]},
              "mmlu": {"history": []}})
    s = compute_standings(lb)
    assert s.miners[0].per_benchmark["math500"] == 0.60   # the 0.99 unmerged attempt is ignored


def test_best_of_multiple_wins_per_miner():
    lb = _lb({"math500": {"history": [_win("bob", 0.60, 1), _win("bob", 0.75, 2), _win("bob", 0.70, 3)]},
              "mmlu": {"history": []}})
    s = compute_standings(lb)
    assert s.miners[0].per_benchmark["math500"] == 0.75   # max over the miner's merged wins


def test_missing_benchmark_counts_as_zero_and_reports_competed():
    lb = _lb({"math500": {"history": [_win("solo", 0.80, 1)]},
              "mmlu": {"history": []}})
    s = compute_standings(lb)
    m = s.miners[0]
    assert m.overall == 0.40 and m.n_competed == 1        # 0.80 on 1 of 2 benches -> 0.40


def test_ranking_tiebreak_by_benchmarks_led():
    # equal overall; the one holding a crown ranks first.
    lb = _lb({"math500": {"best_miner": "x", "history": [_win("x", 0.50, 1), _win("y", 0.50, 2)]},
              "mmlu": {"history": []}})
    s = compute_standings(lb)
    assert s.miners[0].miner == "x" and s.miners[0].benchmarks_led == 1


# --------------------------------------------------------------------------- #
# seed / robustness
# --------------------------------------------------------------------------- #
def test_seed_leaderboard_has_no_miners():
    s = compute_standings(_lb({"math500": {"history": []}, "mmlu": {"history": []}}))
    assert s.benchmarks == ["math500", "mmlu"] and s.miners == [] and s.leader is None


def test_tampered_history_does_not_crash():
    lb = _lb({"math500": {"history": 5},                       # non-list history
              "mmlu": {"history": [7, _win("z", 0.5, 1)]}})    # non-dict entry mixed in
    s = compute_standings(lb)                                  # must not raise
    assert s.miners[0].miner == "z" and s.miners[0].per_benchmark == {"mmlu": 0.5}


def test_non_dict_benchmarks_is_empty():
    assert compute_standings({"benchmarks": 7}).miners == []
    assert compute_standings({}).benchmarks == []


# --------------------------------------------------------------------------- #
# load + render
# --------------------------------------------------------------------------- #
def test_load_standings_from_file(tmp_path):
    p = tmp_path / "leaderboard.json"
    p.write_text(json.dumps(_lb({"math500": {"history": [_win("alice", 0.8, 1)]},
                                 "mmlu": {"history": [_win("alice", 0.9, 2)]}})))
    s = load_standings(p)
    assert s.leader == "alice" and s.miners[0].overall == pytest.approx(0.85)
    # missing file -> empty, never raises
    assert load_standings(tmp_path / "nope.json").miners == []


def test_render_table_and_empty():
    lb = _lb({"math500": {"best_miner": "alice", "history": [_win("alice", 0.8, 1)]},
              "mmlu": {"history": [_win("alice", 0.9, 2)]}})
    md = render(compute_standings(lb))
    assert "competition standings" in md.lower() and "equal-weighted" in md
    assert "| 1 | alice |" in md and "overall leader:** alice" in md
    empty = render(compute_standings(_lb({"math500": {"history": []}})))
    assert empty.strip().endswith("(no miners have won a benchmark yet)_")


def test_to_dict_roundtrips_json():
    lb = _lb({"math500": {"history": [_win("alice", 0.8, 1)]}, "mmlu": {"history": []}})
    d = compute_standings(lb).to_dict()
    assert json.loads(json.dumps(d))["leader"] == "alice"
    assert d["miners"][0]["per_benchmark"] == {"math500": 0.8}
