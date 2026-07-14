#!/usr/bin/env python3
"""Report per-model selective-prediction / risk-coverage from oracle matrices.

Reads the ``oracle_matrix_<bench>.json`` files and reports, per model, the risk-coverage
curve driven by self-consistency confidence (``max(p_hat, 1-p_hat)``): AURC, its gain over
a random-ordering baseline, accuracy at a few coverage levels, and the abstention gain.
This is the offline substrate for the IMPROVEMENTS.md #7 UCCI confidence-cascade idea —
whether abstaining on a model's least self-consistent queries actually buys accuracy.

    python scripts/selective_report.py experiments/final/oracle_matrix_*.json
    python scripts/selective_report.py --root experiments --json selective.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.selective import analyze, render


def _files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/oracle_matrix_*.json", recursive=True))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-model selective-prediction / risk-coverage report.")
    ap.add_argument("files", nargs="*", help="oracle_matrix_<bench>.json file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/oracle_matrix_*.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _files(args.files, args.root)
    if not files:
        print("no oracle_matrix JSONs given (pass files or --root)")
        return
    reports = []
    for f in files:
        try:
            matrix = json.loads(Path(f).read_text())
        except Exception:
            continue
        summary = analyze(matrix)
        print(render(summary))
        reports.append(summary.to_dict())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
