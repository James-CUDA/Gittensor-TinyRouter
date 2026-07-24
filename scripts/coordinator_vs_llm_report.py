#!/usr/bin/env python3
"""Verify SPEC R11 offline: does the trained coordinator beat an LLM-as-coordinator?

Reads a JSON of per-benchmark trained-coordinator (TRINITY) accuracy and the
LLM-as-coordinator baseline accuracy, and reports the margin per benchmark plus
the R11 verdict. Zero API cost. SPEC §6 notes the paper's LLM-as-coordinator
average is 53.76 (Table 8), not the text's 64.14.

Input JSON (``--accuracies``): per benchmark, TRINITY accuracy (``trinity`` or
``trained``) and the LLM-as-coordinator accuracy (``llm_as_coordinator``, or the
aliases ``llm_coordinator`` / ``llm``):

    {"livecodebench": {"trinity": 0.615, "llm_coordinator": 0.52},
     "math500":       {"trinity": 0.88,  "llm_coordinator": 0.70},
     "mmlu":          {"trinity": 0.916, "llm_coordinator": 0.60}}

    python scripts/coordinator_vs_llm_report.py --accuracies r11.json

Exits non-zero when R11 is violated. ``--union`` holds on the equal-weight
union margin instead of requiring every benchmark to hold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.coordinator_vs_llm import analyze_benchmarks, render  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected an object of benchmark -> {{trinity, llm_coordinator}}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    """Print the R11 report; exit non-zero when R11 is violated."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--accuracies", required=True, type=Path,
                    help="JSON of benchmark -> {trinity, llm_coordinator}")
    ap.add_argument("--union", action="store_true",
                    help="hold on the equal-weight union average instead of every benchmark")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    accs = _load(args.accuracies)
    # Call the analyzer with its real signature (tol only) — require_all was never a kwarg (#446).
    report = analyze_benchmarks(accs)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(accs))
    # Default: every comparable benchmark must beat the LLM-as-coordinator.
    # ``--union``: hold when the equal-weight union margin is positive.
    holds = report["union_margin"] > 0.0 if args.union else report["r11_holds"]
    return 0 if holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
