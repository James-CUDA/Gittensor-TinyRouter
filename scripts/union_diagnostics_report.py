#!/usr/bin/env python3
"""Cross-benchmark (equal-weight union) sampling & selective-prediction report.

The competition score is the equal-weighted union of the 3 benchmarks, and
``scripts/union_oracle_report.py`` already aggregates the oracle ceiling that way. The other
two per-model diagnostics had no union view: ``sampling_report.py`` and
``selective_report.py`` are strictly per-matrix, so their headline verdicts — *does
re-sampling the best single model rival the routing oracle?* and *is self-consistency
confidence informative enough to abstain on?* — could only ever be read one benchmark at a
time, which is not the question the composite asks.

    python scripts/union_diagnostics_report.py experiments/final/oracle_matrix_*.json
    python scripts/union_diagnostics_report.py --root experiments --json union_diag.json
    python scripts/union_diagnostics_report.py --root experiments --only sampling

Reuses the canonical ``sampling.analyze`` / ``selective.analyze`` per matrix, so the union
can never disagree with the per-benchmark reports it summarises. Where the per-benchmark
verdicts DISAGREE, that split is called out rather than averaged away.

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from trinity.analysis.union_diagnostics import (
    render_sampling,
    render_selective,
    union_sampling,
    union_selective,
)


def _matrix_files(paths: list[str], root: str | None) -> list[str]:
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
    ap = argparse.ArgumentParser(description="Cross-benchmark union sampling/selective report.")
    ap.add_argument("files", nargs="*", help="oracle_matrix_<bench>.json file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/oracle_matrix_*.json")
    ap.add_argument("--only", choices=("sampling", "selective"), default=None,
                    help="render just one of the two sections")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _matrix_files(args.files, args.root)
    if not files:
        print("no oracle_matrix JSONs given (pass files or --root)")
        return

    matrices = []
    for f in files:
        try:
            matrices.append(json.loads(Path(f).read_text()))
        except Exception:
            continue
    if not matrices:
        print("no readable oracle_matrix JSONs")
        return

    payload: dict = {}
    try:
        if args.only != "selective":
            s = union_sampling(matrices)
            print(render_sampling(s))
            payload["sampling"] = s.to_dict()
        if args.only != "sampling":
            sel = union_selective(matrices)
            print(render_selective(sel))
            payload["selective"] = sel.to_dict()
    except ValueError as exc:
        # A mismatched model set makes an equal-weight per-model average meaningless;
        # say so instead of emitting a number nobody can interpret.
        print(f"cannot form the union: {exc}")
        return

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
