"""CLI over the merged multi-level reachability analysis (ORACLE §2.2 / §6).

``analysis/reachability.py`` shipped in #417 but nothing could invoke it. These
tests drive ``scripts/reachability_report.py`` end to end, and pin the two
strings in ``oracle_ceiling.py`` that used to describe L1/L2 as "future" and
advise widening a level with no tool behind it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "reachability_report.py"
_ORACLE_CEILING = _REPO / "scripts" / "oracle_ceiling.py"
_SRC = str(_REPO / "src")

MODELS = ("m1", "m2")


def _matrix(solved_by_model, benchmark="math500", n=6):
    """Build an ``oracle_matrix`` where each model solves a fixed share of tasks.

    ``solved_by_model`` maps model -> number of leading tasks it solves.
    """
    tasks = []
    for i in range(n):
        per_model = {
            m: [1 if i < solved_by_model.get(m, 0) else 0]
            for m in MODELS
        }
        tasks.append({"id": f"q{i}", "per_model": per_model})
    return {"benchmark": benchmark, "tasks": tasks}


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def _run(*args):
    env = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, env=env
    )


# --------------------------------------------------------------------------
# happy paths
# --------------------------------------------------------------------------


def test_single_level_report_runs(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3, "m2": 2}))
    r = _run("--matrix", f"L0={p}")
    assert r.returncode == 0, r.stderr
    assert "L0" in r.stdout


def test_two_levels_are_both_reported(tmp_path):
    l0 = _write(tmp_path, "l0.json", _matrix({"m1": 2, "m2": 1}))
    l1 = _write(tmp_path, "l1.json", _matrix({"m1": 4, "m2": 3}))
    r = _run("--matrix", f"L0={l0}", "--matrix", f"L1={l1}")
    assert r.returncode == 0, r.stderr
    assert "L0" in r.stdout and "L1" in r.stdout


def test_level_argument_is_case_insensitive(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    r = _run("--matrix", f"l0={p}")
    assert r.returncode == 0, r.stderr


def test_json_output_is_parseable(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3, "m2": 2}))
    r = _run("--matrix", f"L0={p}", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert "verdict" in payload


def test_threshold_is_accepted(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    r = _run("--matrix", f"L0={p}", "--threshold", "0.9")
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------
# the monotonicity guard -- the reason multi-level analysis needs a guard
# --------------------------------------------------------------------------


def test_a_narrower_level_beating_a_wider_one_is_inconsistent(tmp_path):
    """Oracle must be non-decreasing in reachability; L0 > L1 means bad data."""
    l0 = _write(tmp_path, "l0.json", _matrix({"m1": 6, "m2": 6}))   # everything solved
    l1 = _write(tmp_path, "l1.json", _matrix({"m1": 0, "m2": 0}))   # nothing solved
    r = _run("--matrix", f"L0={l0}", "--matrix", f"L1={l1}")
    assert r.returncode == 1
    assert "INCONSISTENT" in r.stdout or "INCONSISTENT" in r.stderr


def test_monotone_levels_do_not_trip_the_guard(tmp_path):
    l0 = _write(tmp_path, "l0.json", _matrix({"m1": 1, "m2": 1}))
    l1 = _write(tmp_path, "l1.json", _matrix({"m1": 5, "m2": 5}))
    r = _run("--matrix", f"L0={l0}", "--matrix", f"L1={l1}")
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------
# confidence intervals
# --------------------------------------------------------------------------


def test_cis_are_accepted(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3, "m2": 2}))
    cis = _write(tmp_path, "cis.json", {"L0": [0.0, 0.01]})
    r = _run("--matrix", f"L0={p}", "--cis", str(cis), "--json")
    assert r.returncode == 0, r.stderr
    assert "verdict" in json.loads(r.stdout)


def test_thin_headroom_at_l0_only_is_a_lower_bound_not_an_all_clear(tmp_path):
    """ORACLE §2.2's core point: a thin L0 gap does NOT rule routing out.

    ``_matrix`` gives nested solve sets, so the routing oracle equals the best
    single model and headroom is 0 — thin. Because L0 is the narrowest level,
    that must read as LOWER_BOUND_ONLY, not as "routing cannot help".
    """
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3, "m2": 2}))
    r = _run("--matrix", f"L0={p}", "--json")
    payload = json.loads(r.stdout)
    assert payload["verdict"] == "LOWER_BOUND_ONLY"
    assert payload["can_rule_out_routing"] is False


def test_a_sampled_widest_level_without_a_ci_cannot_carry_a_verdict(tmp_path):
    """ORACLE §2.2: a sampled probe must be reported with its own CI."""
    l2 = _write(tmp_path, "l2.json", _matrix({"m1": 3, "m2": 2}))
    payload = json.loads(_run("--matrix", f"L2={l2}", "--json").stdout)
    assert payload["verdict"] == "NEEDS_CI"
    assert payload["can_rule_out_routing"] is False


def test_thin_headroom_at_the_widest_level_with_a_ci_rules_routing_out(tmp_path):
    """The same thin gap at L2 — widest level defined — settles it once a CI backs it."""
    l2 = _write(tmp_path, "l2.json", _matrix({"m1": 3, "m2": 2}))
    cis = _write(tmp_path, "cis.json", {"L2": [0.0, 0.01]})
    payload = json.loads(
        _run("--matrix", f"L2={l2}", "--cis", str(cis), "--json").stdout
    )
    assert payload["verdict"] == "POOL_BOUND"
    assert payload["can_rule_out_routing"] is True


def test_supplying_a_ci_is_what_moves_a_sampled_level_off_needs_ci(tmp_path):
    """The CI is load-bearing, not decoration."""
    l2 = _write(tmp_path, "l2.json", _matrix({"m1": 3, "m2": 2}))
    cis = _write(tmp_path, "cis.json", {"L2": [0.0, 0.01]})
    without = json.loads(_run("--matrix", f"L2={l2}", "--json").stdout)["verdict"]
    with_ci = json.loads(
        _run("--matrix", f"L2={l2}", "--cis", str(cis), "--json").stdout
    )["verdict"]
    assert without == "NEEDS_CI" and with_ci != "NEEDS_CI"


def test_inverted_ci_is_rejected(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    cis = _write(tmp_path, "cis.json", {"L0": [0.5, 0.1]})
    r = _run("--matrix", f"L0={p}", "--cis", str(cis))
    assert r.returncode == 2
    assert "inverted" in r.stderr


def test_ci_for_an_unknown_level_is_rejected(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    cis = _write(tmp_path, "cis.json", {"L9": [0.0, 0.1]})
    r = _run("--matrix", f"L0={p}", "--cis", str(cis))
    assert r.returncode == 2


def test_malformed_ci_shape_is_rejected(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    cis = _write(tmp_path, "cis.json", {"L0": [0.1]})
    r = _run("--matrix", f"L0={p}", "--cis", str(cis))
    assert r.returncode == 2
    assert "[lo, hi]" in r.stderr


def test_missing_ci_file_is_graceful(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    r = _run("--matrix", f"L0={p}", "--cis", str(tmp_path / "nope.json"))
    assert r.returncode == 2
    assert "no such file" in r.stderr


# --------------------------------------------------------------------------
# usage errors
# --------------------------------------------------------------------------


def test_no_matrices_is_a_usage_error():
    r = _run()
    assert r.returncode == 2
    assert "at least one --matrix" in r.stderr


def test_unknown_level_is_rejected(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    r = _run("--matrix", f"L7={p}")
    assert r.returncode == 2
    assert "unknown level" in r.stderr


def test_missing_equals_is_rejected(tmp_path):
    r = _run("--matrix", "justapath.json")
    assert r.returncode == 2
    assert "LEVEL=PATH" in r.stderr


def test_duplicate_level_is_rejected(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    r = _run("--matrix", f"L0={p}", "--matrix", f"L0={p}")
    assert r.returncode == 2
    assert "more than once" in r.stderr


def test_missing_matrix_file_is_graceful(tmp_path):
    r = _run("--matrix", f"L0={tmp_path / 'nope.json'}")
    assert r.returncode == 2
    assert "no such file" in r.stderr


def test_malformed_matrix_json_is_graceful(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    r = _run("--matrix", f"L0={p}")
    assert r.returncode == 2
    assert "bad input" in r.stderr


def test_scalar_matrix_payload_is_rejected(tmp_path):
    p = _write(tmp_path, "scalar.json", 42)
    r = _run("--matrix", f"L0={p}")
    assert r.returncode == 2


def test_empty_path_after_equals_is_rejected():
    r = _run("--matrix", "L0=")
    assert r.returncode == 2
    assert "no path" in r.stderr


# --------------------------------------------------------------------------
# the mislabelling warning
# --------------------------------------------------------------------------


def test_a_wider_level_prints_the_collection_caveat(tmp_path):
    """Collection does not branch on level, so a wider label is user-asserted."""
    l0 = _write(tmp_path, "l0.json", _matrix({"m1": 1}))
    l1 = _write(tmp_path, "l1.json", _matrix({"m1": 4}))
    r = _run("--matrix", f"L0={l0}", "--matrix", f"L1={l1}")
    assert "does not itself branch on level" in r.stdout


def test_l0_only_does_not_print_the_caveat(tmp_path):
    p = _write(tmp_path, "l0.json", _matrix({"m1": 3}))
    r = _run("--matrix", f"L0={p}")
    assert "does not itself branch on level" not in r.stdout


# --------------------------------------------------------------------------
# oracle_ceiling.py's stale strings
# --------------------------------------------------------------------------


def test_collector_level_flag_is_still_l0_only():
    """Collection genuinely cannot do L1/L2; the flag must not claim otherwise."""
    text = _ORACLE_CEILING.read_text()
    assert 'choices=["L0"]' in text


def test_collector_no_longer_calls_l1_l2_future():
    """The analysis shipped in #417; the help text said 'are future'."""
    assert "L1/L2 are future" not in _ORACLE_CEILING.read_text()


def test_inconclusive_hint_points_at_a_tool_that_exists():
    text = _ORACLE_CEILING.read_text()
    assert "reachability_report.py" in text


def test_the_referenced_script_exists():
    assert _SCRIPT.exists()


@pytest.mark.parametrize("flag", ["--matrix", "--cis", "--threshold", "--json"])
def test_documented_flags_are_real(flag):
    r = _run("--help")
    assert r.returncode == 0
    assert flag in r.stdout
