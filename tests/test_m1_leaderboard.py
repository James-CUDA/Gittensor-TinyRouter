"""King / challenger decision + leaderboard I/O (no GPU)."""
from __future__ import annotations

from pathlib import Path

from trinity.m1.leaderboard import (
    decide_m1_winner,
    load_m1_leaderboard,
    promote_king,
    save_m1_leaderboard,
)
from trinity.m1.metrics import TriageMetrics
from trinity.m1.scoring import compare_report


def test_no_king_challenger_wins():
    ok, reason = decide_m1_winner(king_composite=None, challenger_composite=0.1)
    assert ok
    assert "no_king" in reason


def test_margin_required():
    ok, _ = decide_m1_winner(king_composite=0.50, challenger_composite=0.51, win_margin=0.02)
    assert not ok
    ok2, _ = decide_m1_winner(king_composite=0.50, challenger_composite=0.52, win_margin=0.02)
    assert ok2


def test_promote_and_reload(tmp_path: Path):
    lb = load_m1_leaderboard(tmp_path, "5-domain")
    assert lb.king is None
    promote_king(
        lb,
        miner="alice",
        submission="submissions/alice/m1",
        composite=0.55,
        metrics={"composite": 0.55},
    )
    path = save_m1_leaderboard(tmp_path, lb)
    assert path.exists()
    lb2 = load_m1_leaderboard(tmp_path, "5-domain")
    assert lb2.king is not None
    assert lb2.king.miner == "alice"
    assert lb2.king.composite == 0.55
    assert len(lb2.history) == 1


def test_compare_report_merge_flag():
    king = TriageMetrics(
        n=10,
        domain_accuracy=0.5,
        domain_macro_f1=0.5,
        difficulty_exact=0.5,
        difficulty_within_1=0.9,
        joint_accuracy=0.4,
        composite=0.50,
        per_domain_accuracy={},
    )
    chall = TriageMetrics(
        n=10,
        domain_accuracy=0.8,
        domain_macro_f1=0.7,
        difficulty_exact=0.6,
        difficulty_within_1=0.95,
        joint_accuracy=0.5,
        composite=0.74,
        per_domain_accuracy={},
    )
    rep = compare_report(
        king_metrics=king,
        challenger_metrics=chall,
        win_margin=0.02,
        king_meta={"miner": "alice"},
        challenger_meta={"miner": "bob"},
    )
    assert rep["merge"] is True
    assert rep["challenger"]["miner"] == "bob"
