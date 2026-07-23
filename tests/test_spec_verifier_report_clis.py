"""Tests for the SPEC-verifier report CLIs (scripts/*_report.py).

Every analysis module ships a ``scripts/<name>_report.py`` CLI; the R3/R4/R8/R13/R1R2
verifier modules were the only ones missing theirs. These drive each CLI's ``main()``
in-process (no torch, no network, no subprocess) and assert it runs, prints the module's
report, and returns 0 (invariant holds) / 1 (violated).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_cli(name: str):
    path = _REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, name, arg, data, extra=None):
    p = tmp_path / "in.json"
    p.write_text(json.dumps(data))
    cli = _load_cli(name)
    return cli.main([arg, str(p), *(extra or [])])


# --------------------------------------------------------------------------- #
# R8 — optimizer_ranking
# --------------------------------------------------------------------------- #
def test_r8_report_holds_and_violation(tmp_path):
    ok = {"math500": {"sep_cmaes": 0.72, "sft": 0.66, "rs": 0.60, "reinforce": 0.55}}
    bad = {"math500": {"sep_cmaes": 0.50, "sft": 0.66, "rs": 0.60, "reinforce": 0.55}}
    assert _run(tmp_path, "optimizer_ranking_report", "--scores", ok) == 0
    assert _run(tmp_path, "optimizer_ranking_report", "--scores", bad) == 1


# --------------------------------------------------------------------------- #
# R3 — multi_agent_baseline
# --------------------------------------------------------------------------- #
def test_r3_report_holds_and_violation(tmp_path):
    ok = {"math500": {"trinity": 0.88, "baselines": {"MoA": 0.66, "MasRouter": 0.68}}}
    bad = {"math500": {"trinity": 0.60, "baselines": {"MasRouter": 0.70}}}
    assert _run(tmp_path, "multi_agent_baseline_report", "--accuracies", ok) == 0
    assert _run(tmp_path, "multi_agent_baseline_report", "--accuracies", bad) == 1


# --------------------------------------------------------------------------- #
# R4 — random_routing
# --------------------------------------------------------------------------- #
def test_r4_report_holds_and_violation(tmp_path):
    ok = {"rlpr": {"trinity": 0.41, "random_routing": 0.32}}
    bad = {"mmlu": {"trinity": 0.27, "random_routing": 0.28}}
    assert _run(tmp_path, "random_routing_report", "--accuracies", ok) == 0
    assert _run(tmp_path, "random_routing_report", "--accuracies", bad) == 1


# --------------------------------------------------------------------------- #
# R13 — relative_error_reduction
# --------------------------------------------------------------------------- #
def test_r13_report_holds_and_violation(tmp_path):
    ok = {"math500": {"trinity": 0.88, "best_single": 0.80}}
    bad = {"rlpr": {"trinity": 0.40, "best_single": 0.45}}
    assert _run(tmp_path, "relative_error_reduction_report", "--accuracies", ok) == 0
    assert _run(tmp_path, "relative_error_reduction_report", "--accuracies", bad) == 1


def test_r13_report_accepts_singles_map(tmp_path):
    data = {"mmlu": {"trinity": 0.92, "singles": {"gpt5": 0.90, "gemini": 0.88}}}
    assert _run(tmp_path, "relative_error_reduction_report", "--accuracies", data) == 0


# --------------------------------------------------------------------------- #
# R1/R2 — single_model_dominance
# --------------------------------------------------------------------------- #
def test_r1r2_report_combined_and_flags(tmp_path):
    # R2 fails on one task but R1 (average) still holds.
    data = {
        "math500": {"trinity": 0.95, "singles": {"gpt5": 0.70}},
        "livecodebench": {"trinity": 0.60, "singles": {"gpt5": 0.65}},   # R2 violation
    }
    assert _run(tmp_path, "single_model_dominance_report", "--accuracies", data) == 1  # combined
    assert _run(tmp_path, "single_model_dominance_report", "--accuracies", data, ["--r2-only"]) == 1
    assert _run(tmp_path, "single_model_dominance_report", "--accuracies", data, ["--r1-only"]) == 0


def test_reports_json_flag(tmp_path, capsys):
    ok = {"rlpr": {"trinity": 0.41, "random_routing": 0.32}}
    _run(tmp_path, "random_routing_report", "--accuracies", ok, ["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["r4_holds"] is True
