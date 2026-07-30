"""Regression test for the R11 report CLI (scripts/coordinator_vs_llm_report.py).

The script shipped broken: it called ``analyze_benchmarks(accs, require_all=...)`` and
``render(accs, require_all=...)``, but the module exposes neither keyword — so every
invocation raised ``TypeError: analyze_benchmarks() got an unexpected keyword argument
'require_all'``. It also documented the input key ``llm_coordinator`` while the analyzer
reads ``llm_as_coordinator``. This drives the CLI's ``main()`` in-process (no torch, no
network) and asserts it now runs and returns the right exit code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "coordinator_vs_llm_report.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("coordinator_vs_llm_report", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["coordinator_vs_llm_report"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "r11.json"
    p.write_text(json.dumps(data))
    return p


def test_cli_runs_and_reports_hold(tmp_path, capsys):
    cli = _load_cli()
    path = _write(tmp_path, {
        "math500": {"trinity": 0.88, "llm_coordinator": 0.70},
        "mmlu": {"trinity": 0.916, "llm_coordinator": 0.60},
    })
    rc = cli.main(["--accuracies", str(path)])
    assert rc == 0                                   # R11 holds -> exit 0
    assert "R11 (trained coordinator > LLM-as-coordinator): HOLDS" in capsys.readouterr().out


def test_cli_exit_nonzero_on_violation(tmp_path):
    cli = _load_cli()
    path = _write(tmp_path, {"mmlu": {"trinity": 0.50, "llm_coordinator": 0.60}})
    assert cli.main(["--accuracies", str(path)]) == 1


def test_cli_accepts_both_llm_key_aliases(tmp_path):
    cli = _load_cli()
    # documented alias `llm_coordinator` and the module's `llm_as_coordinator` both work
    p1 = _write(tmp_path, {"math500": {"trinity": 0.88, "llm_coordinator": 0.70}})
    assert cli.main(["--accuracies", str(p1)]) == 0
    p2 = tmp_path / "r11b.json"
    p2.write_text(json.dumps({"math500": {"trinity": 0.88, "llm_as_coordinator": 0.70}}))
    assert cli.main(["--accuracies", str(p2)]) == 0


def test_cli_json_output_is_valid(tmp_path, capsys):
    cli = _load_cli()
    path = _write(tmp_path, {"math500": {"trinity": 0.88, "llm_coordinator": 0.70}})
    cli.main(["--accuracies", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["r11_holds"] is True
    assert payload["union_margin"] == pytest.approx(0.18)


def test_cli_union_flag_uses_union_margin(tmp_path):
    cli = _load_cli()
    # A benchmark that loses individually but the union average still wins.
    path = _write(tmp_path, {
        "math500": {"trinity": 0.95, "llm_coordinator": 0.60},
        "mmlu": {"trinity": 0.55, "llm_coordinator": 0.60},   # loses here
    })
    assert cli.main(["--accuracies", str(path)]) == 1            # default: every bench
    assert cli.main(["--accuracies", str(path), "--union"]) == 0  # union avg wins
