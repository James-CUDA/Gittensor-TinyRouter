"""Transcript-budget diagnostic (SPEC §4.5 truncation cost).

Fixtures are produced by the **real** ``roles.postprocess.postprocess`` rather
than by hand-written "truncated" strings, so the diagnostic is validated against
the function it measures. If the truncation policy changes, these tests move with
it instead of silently measuring a stale format.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trinity.analysis.transcript_budget import (
    DEFAULT_CONTEXT_TOKENS,
    RoleBudget,
    TranscriptBudget,
    VerdictLoss,
    analyze,
    analyze_benchmarks,
    counts_by_role,
    render,
)
from trinity.roles.postprocess import ELISION_MARKER, postprocess
from trinity.roles.verifier import parse_verdict
from trinity.types import Role

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transcript_budget_report.py"


def _turn(role, raw, *, max_chars=8000):
    """A turn record whose processed_output comes from the real postprocess."""
    return {
        "role": role,
        "raw_output": raw,
        "processed_output": postprocess(raw, role, max_chars=max_chars),
    }


def _long(marker: str, n: int = 400) -> str:
    return f"{marker} " + ("x" * n)


# --------------------------------------------------------------------------
# RoleBudget arithmetic
# --------------------------------------------------------------------------


def test_rates_are_zero_on_an_empty_budget():
    b = RoleBudget(role="worker")
    assert b.truncation_rate == 0.0
    assert b.elision_rate == 0.0
    assert b.elided_chars == 0


def test_elided_chars_never_negative():
    """Stripping can make kept longer than a naive raw count; clamp at zero."""
    b = RoleBudget(role="worker", turns=1, raw_chars=10, kept_chars=25)
    assert b.elided_chars == 0


def test_rates_are_computed_per_role():
    b = RoleBudget(role="worker", turns=4, truncated=1, raw_chars=100, kept_chars=60)
    assert b.truncation_rate == pytest.approx(0.25)
    assert b.elided_chars == 40
    assert b.elision_rate == pytest.approx(0.4)


# --------------------------------------------------------------------------
# truncation detection
# --------------------------------------------------------------------------


def test_short_outputs_are_not_flagged_as_truncated():
    rep = analyze([_turn(Role.WORKER, "42")])
    assert rep.pooled.truncated == 0
    assert rep.pooled.turns == 1


def test_whitespace_stripping_alone_is_not_truncation():
    """postprocess strips; that must not be mistaken for elision."""
    rep = analyze([_turn(Role.WORKER, "   the answer is 42   ")])
    assert rep.pooled.truncated == 0
    assert rep.pooled.elided_chars == 0


def test_a_real_truncation_is_detected():
    rep = analyze([_turn(Role.WORKER, _long("answer", 5000), max_chars=200)])
    assert rep.pooled.truncated == 1
    assert rep.pooled.elided_chars > 0
    assert rep.pooled.truncation_rate == pytest.approx(1.0)


def test_hard_head_truncation_without_a_marker_is_still_detected():
    """When the budget is smaller than the marker, postprocess emits no marker."""
    raw = _long("answer", 500)
    kept = postprocess(raw, Role.WORKER, max_chars=5)
    assert ELISION_MARKER not in kept
    rep = analyze([{"role": "worker", "raw_output": raw, "processed_output": kept}])
    assert rep.pooled.truncated == 1


def test_max_raw_chars_tracks_the_longest_output():
    rep = analyze([
        _turn(Role.WORKER, "short"),
        _turn(Role.WORKER, _long("long", 900), max_chars=100),
    ])
    assert rep.pooled.max_raw_chars > 900


# --------------------------------------------------------------------------
# verdict survival -- the claim postprocess makes about its tail bias
# --------------------------------------------------------------------------


def _verdict_loss_fixture(max_chars=300):
    """A verifier whose committed REVISE sits mid-output behind an earlier ACCEPT.

    ``parse_verdict`` takes the LAST match, so the raw output commits REVISE.
    Truncation elides the middle, leaving only the earlier ACCEPT in the head --
    exactly the failure the tail bias is supposed to prevent but cannot here.
    """
    raw = (
        "VERDICT: ACCEPT looks plausible at first glance\n"
        + ("filler " * 60)
        + "\nVERDICT: REVISE the derivation is wrong\n"
        + ("tail " * 60)
    )
    return raw, postprocess(raw, Role.VERIFIER, max_chars=max_chars)


def test_a_lost_verdict_is_reported():
    raw, kept = _verdict_loss_fixture()
    assert parse_verdict(raw) == "REVISE"
    assert parse_verdict(kept) != "REVISE"  # the fixture must actually lose it
    rep = analyze([{"role": "verifier", "raw_output": raw, "processed_output": kept}])
    assert len(rep.verdict_losses) == 1
    loss = rep.verdict_losses[0]
    assert isinstance(loss, VerdictLoss)
    assert loss.raw_verdict == "REVISE"
    assert loss.kept_verdict != "REVISE"
    assert loss.index == 0


def test_a_verdict_at_the_tail_survives_truncation():
    """The tail bias working as designed -- must NOT be reported as a loss."""
    raw = ("filler " * 400) + "\nVERDICT: ACCEPT\n"
    kept = postprocess(raw, Role.VERIFIER, max_chars=200)
    assert parse_verdict(kept) == "ACCEPT"
    rep = analyze([{"role": "verifier", "raw_output": raw, "processed_output": kept}])
    assert rep.pooled.truncated == 1
    assert rep.verdict_losses == ()


def test_untruncated_verifier_turns_are_never_verdict_losses():
    """A missing verdict on an untruncated turn is a model issue, not truncation."""
    rep = analyze([_turn(Role.VERIFIER, "I have no opinion")])
    assert rep.pooled.truncated == 0
    assert rep.verdict_losses == ()


def test_a_truncated_worker_turn_is_not_checked_for_verdicts():
    rep = analyze([_turn(Role.WORKER, _long("VERDICT: ACCEPT", 3000), max_chars=120)])
    assert rep.pooled.truncated == 1
    assert rep.verdict_losses == ()


def test_verdict_losses_carry_the_turn_index():
    raw, kept = _verdict_loss_fixture()
    turns = [_turn(Role.WORKER, "fine"), {"role": "verifier", "raw_output": raw,
                                          "processed_output": kept}]
    rep = analyze(turns)
    assert [v.index for v in rep.verdict_losses] == [1]


# --------------------------------------------------------------------------
# per-role split
# --------------------------------------------------------------------------


def test_per_role_split_counts_each_role_separately():
    rep = analyze([
        _turn(Role.THINKER, "plan"),
        _turn(Role.WORKER, "work"),
        _turn(Role.WORKER, "more work"),
        _turn(Role.VERIFIER, "VERDICT: ACCEPT"),
    ])
    assert rep.per_role["worker"].turns == 2
    assert rep.per_role["thinker"].turns == 1
    assert rep.per_role["verifier"].turns == 1
    assert rep.pooled.turns == 4


def test_role_enum_and_string_are_normalized_the_same():
    a = analyze([{"role": Role.WORKER, "raw_output": "x", "processed_output": "x"}])
    b = analyze([{"role": "  Worker ", "raw_output": "x", "processed_output": "x"}])
    assert set(a.per_role) == set(b.per_role) == {"worker"}


def test_raw_and_processed_aliases_are_accepted():
    rep = analyze([{"role": "worker", "raw": "hello there", "processed": "hello"}])
    assert rep.pooled.raw_chars == len("hello there")
    assert rep.pooled.kept_chars == len("hello")


def test_counts_by_role_helper():
    turns = [_turn(Role.WORKER, "a"), _turn(Role.WORKER, "b"), _turn(Role.THINKER, "c")]
    assert counts_by_role(turns) == {"worker": 2, "thinker": 1}


def test_empty_input_is_graceful():
    rep = analyze([])
    assert rep.pooled.turns == 0
    assert rep.per_role == {}
    assert rep.est_transcript_tokens == 0
    assert rep.revisit_recommended is False


# --------------------------------------------------------------------------
# context overflow -- SPEC 4.5's revisit trigger
# --------------------------------------------------------------------------


def test_no_overflow_on_a_small_transcript():
    rep = analyze([_turn(Role.WORKER, "42")])
    assert rep.overflows_context is False
    assert rep.revisit_recommended is False
    assert rep.context_headroom_tokens > 0


def test_overflow_sets_the_revisit_flag():
    big = "y" * 40_000
    rep = analyze(
        [{"role": "worker", "raw_output": big, "processed_output": big}],
        context_tokens=100,
    )
    assert rep.overflows_context is True
    assert rep.revisit_recommended is True
    assert rep.context_headroom_tokens < 0


def test_token_estimate_uses_the_chars_per_token_divisor():
    text = "z" * 400
    rep = analyze(
        [{"role": "worker", "raw_output": text, "processed_output": text}],
        chars_per_token=4.0,
    )
    assert rep.est_transcript_tokens == 100


def test_default_context_window_is_the_encoder_context():
    rep = analyze([])
    assert rep.context_tokens == DEFAULT_CONTEXT_TOKENS


def test_invalid_chars_per_token_rejected():
    with pytest.raises(ValueError, match="chars_per_token"):
        analyze([], chars_per_token=0)


def test_negative_context_tokens_rejected():
    with pytest.raises(ValueError, match="context_tokens"):
        analyze([], context_tokens=-1)


# --------------------------------------------------------------------------
# analyze_benchmarks
# --------------------------------------------------------------------------


def test_analyze_benchmarks_reports_each_plus_pooled():
    out = analyze_benchmarks({
        "math500": [_turn(Role.WORKER, "a")],
        "drop": [_turn(Role.WORKER, "b"), _turn(Role.THINKER, "c")],
    })
    assert set(out) == {"math500", "drop", "all"}
    assert out["all"].pooled.turns == 3


def test_pooled_entry_is_turn_weighted_not_an_average_of_rates():
    """One benchmark with many clean turns must dominate one with a single dirty turn."""
    clean = [_turn(Role.WORKER, "ok") for _ in range(9)]
    dirty = [_turn(Role.WORKER, _long("big", 4000), max_chars=100)]
    out = analyze_benchmarks({"clean": clean, "dirty": dirty})
    assert out["dirty"].pooled.truncation_rate == pytest.approx(1.0)
    assert out["all"].pooled.truncation_rate == pytest.approx(0.1)


def test_analyze_benchmarks_on_empty_mapping():
    assert analyze_benchmarks({}) == {}


def test_analyze_benchmarks_accepts_iterators():
    out = analyze_benchmarks({"a": iter([_turn(Role.WORKER, "x")])})
    assert out["a"].pooled.turns == 1
    assert out["all"].pooled.turns == 1


# --------------------------------------------------------------------------
# to_dict / render
# --------------------------------------------------------------------------


def test_to_dict_is_json_serializable_and_complete():
    raw, kept = _verdict_loss_fixture()
    rep = analyze([{"role": "verifier", "raw_output": raw, "processed_output": kept}])
    d = rep.to_dict()
    json.dumps(d)
    assert d["n_verdict_losses"] == 1
    assert set(d) >= {"pooled", "per_role", "verdict_losses", "overflows_context",
                      "revisit_recommended", "context_headroom_tokens", "max_chars"}


def test_render_reports_a_clean_run():
    out = render([_turn(Role.WORKER, "42")])
    assert "verdict losses : none" in out
    assert "no overflow" in out


def test_render_calls_out_verdict_losses():
    raw, kept = _verdict_loss_fixture()
    out = render([{"role": "verifier", "raw_output": raw, "processed_output": kept}])
    assert "VERDICT LOSSES" in out
    assert "REVISE" in out


def test_render_calls_out_overflow():
    big = "y" * 40_000
    out = render([{"role": "worker", "raw_output": big, "processed_output": big}],
                 context_tokens=10)
    assert "REVISIT" in out


def test_render_accepts_a_precomputed_report():
    rep = analyze([_turn(Role.WORKER, "42")])
    assert render(report=rep) == render([_turn(Role.WORKER, "42")])


def test_render_rejects_both_or_neither():
    rep = analyze([])
    with pytest.raises(ValueError, match="exactly one"):
        render([], report=rep)
    with pytest.raises(ValueError, match="exactly one"):
        render()


def test_render_lists_unknown_roles_too():
    out = render([{"role": "narrator", "raw_output": "x", "processed_output": "x"}])
    assert "narrator" in out


# --------------------------------------------------------------------------
# report script
# --------------------------------------------------------------------------


def _run_script(*args):
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, env=env
    )


def test_script_reads_a_flat_turn_list(tmp_path):
    p = tmp_path / "turns.json"
    p.write_text(json.dumps([_turn(Role.WORKER, "42")]))
    r = _run_script("--turns", str(p))
    assert r.returncode == 0, r.stderr
    assert "transcript budget" in r.stdout


def test_script_reads_a_benchmark_mapping(tmp_path):
    p = tmp_path / "turns.json"
    p.write_text(json.dumps({"math500": [_turn(Role.WORKER, "42")]}))
    r = _run_script("--turns", str(p))
    assert r.returncode == 0, r.stderr
    assert "== math500 ==" in r.stdout
    assert "== all ==" in r.stdout


def test_script_json_flag_emits_parseable_json(tmp_path):
    p = tmp_path / "turns.json"
    p.write_text(json.dumps([_turn(Role.WORKER, "42")]))
    r = _run_script("--turns", str(p), "--json")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["pooled"]["turns"] == 1


def test_script_exits_nonzero_on_a_verdict_loss(tmp_path):
    raw, kept = _verdict_loss_fixture()
    p = tmp_path / "turns.json"
    p.write_text(json.dumps([{"role": "verifier", "raw_output": raw,
                              "processed_output": kept}]))
    r = _run_script("--turns", str(p))
    assert r.returncode == 1
    assert "VERDICT LOSSES" in r.stdout


def test_script_exits_nonzero_on_overflow(tmp_path):
    big = "y" * 40_000
    p = tmp_path / "turns.json"
    p.write_text(json.dumps([{"role": "worker", "raw_output": big,
                              "processed_output": big}]))
    r = _run_script("--turns", str(p), "--context-tokens", "10")
    assert r.returncode == 1


def test_script_missing_file_is_graceful(tmp_path):
    r = _run_script("--turns", str(tmp_path / "nope.json"))
    assert r.returncode == 2
    assert "no such file" in r.stderr


def test_script_rejects_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json at all")
    r = _run_script("--turns", str(p))
    assert r.returncode == 2
    assert "could not read" in r.stderr


def test_script_rejects_a_scalar_payload(tmp_path):
    p = tmp_path / "scalar.json"
    p.write_text("42")
    r = _run_script("--turns", str(p))
    assert r.returncode == 2


# --------------------------------------------------------------------------
# import cost
# --------------------------------------------------------------------------


def test_module_imports_without_torch():
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import sys; import trinity.analysis.transcript_budget; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, env=env)
    assert out.stdout.strip() == "False", out.stdout


def test_dataclasses_are_frozen():
    """The report is a value object; callers must not mutate a shared result."""
    rep = analyze([_turn(Role.WORKER, "x")])
    assert isinstance(rep, TranscriptBudget)
    with pytest.raises(Exception):
        rep.pooled.turns = 99  # type: ignore[misc]
