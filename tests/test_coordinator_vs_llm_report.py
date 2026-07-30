"""Regression test for the R11 report CLI (scripts/coordinator_vs_llm_report.py).

The script shipped broken: it called ``analyze_benchmarks(accs, require_all=...)`` and
``render(accs, require_all=...)``, but the module exposes neither keyword — so every
invocation raised ``TypeError``. It also documented the input key ``llm_coordinator``
while the analyzer originally only read ``llm_as_coordinator``. This drives the CLI's
``main()`` in-process (no torch, no network) and asserts it now runs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

_SCRIPT = _REPO / "scripts" / "coordinator_vs_llm_report.py"
_spec = importlib.util.spec_from_file_location("coordinator_vs_llm_report", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["coordinator_vs_llm_report"] = _mod
_spec.loader.exec_module(_mod)
main = _mod.main


def _write(tmp: Path, payload: dict) -> Path:
    p = tmp / "r11.json"
    p.write_text(json.dumps(payload))
    return p


def test_cli_runs_with_documented_llm_coordinator_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "math500": {"trinity": 0.88, "llm_coordinator": 0.70},
            "mmlu": {"trinity": 0.90, "llm_coordinator": 0.60},
        },
    )
    assert main(["--accuracies", str(path)]) == 0


def test_cli_accepts_llm_as_coordinator_canonical_key(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"math500": {"trained": 0.80, "llm_as_coordinator": 0.50}},
    )
    assert main(["--accuracies", str(path)]) == 0


def test_cli_exits_nonzero_when_r11_violated(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"math500": {"trinity": 0.40, "llm_coordinator": 0.70}},
    )
    assert main(["--accuracies", str(path)]) == 1


def test_union_holds_on_positive_union_margin_despite_one_loss(tmp_path: Path) -> None:
    # One loss, one big win -> per-bench R11 fails, but union margin can still be > 0.
    path = _write(
        tmp_path,
        {
            "math500": {"trinity": 0.40, "llm_coordinator": 0.50},  # loss
            "mmlu": {"trinity": 0.95, "llm_coordinator": 0.50},  # big win
        },
    )
    assert main(["--accuracies", str(path)]) == 1  # default: every-bench
    assert main(["--accuracies", str(path), "--union"]) == 0


def test_analyzer_reads_llm_coordinator_alias_directly() -> None:
    from trinity.analysis.coordinator_vs_llm import analyze_benchmarks

    report = analyze_benchmarks(
        {"math500": {"trinity": 0.9, "llm_coordinator": 0.5}}
    )
    assert report["r11_holds"] is True
    assert report["per_benchmark"][0]["comparable"] is True
