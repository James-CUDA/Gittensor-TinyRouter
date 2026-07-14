#!/usr/bin/env python3
"""Reconcile ``oracle_matrix_<bench>.json`` against ``<bench>_rigorous.json`` (integrity guard).

``docs/ORACLE_CEILING_DIAGNOSTIC.md`` §5.3 mandates cross-checking the collected per-query
matrix — the input every oracle-ceiling / union-oracle / sampling / complementarity
conclusion rests on — against the independent rigorous eval: "the per-query matrix,
averaged per model, must reproduce the rigorous eval numbers ... within CI. If it does
not, the matrix collection is buggy and the verdict is void." ``scripts/oracle_ceiling.py``
never does this. This CLI does, emitting ``TRUSTWORTHY / SUSPECT / VOID``.

    # Verify every pair under experiments/final (CI-usable: exits non-zero on VOID):
    python scripts/reconcile_matrix.py

    # One explicit pair; a status report (always exit 0); or a JSON dump:
    python scripts/reconcile_matrix.py --matrix M.json --rigorous R.json
    python scripts/reconcile_matrix.py --report
    python scripts/reconcile_matrix.py --json out.json

Read-only, pure/offline (no torch, no network, no GPU). Pairs each matrix to its rigorous
file by their in-file ``benchmark`` field, so it is robust to the filename mismatch
(``oracle_matrix_math500`` <-> ``math_rigorous``, both ``benchmark="math500"``). Use
``--strict`` to also fail on SUSPECT.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.analysis.reconcile import (  # noqa: E402  (needs the sys.path insert)
    SUSPECT,
    VOID,
    reconcile,
    render,
)


def _load(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _pairs(root: str, matrix_arg: str | None, rigorous_arg: str | None) -> list[tuple[str, str]]:
    """Pair matrix files to rigorous files by their in-file ``benchmark`` field."""
    if matrix_arg and rigorous_arg:
        return [(matrix_arg, rigorous_arg)]
    matrices = sorted(glob.glob(f"{root}/**/oracle_matrix_*.json", recursive=True))
    rig_by_bench: dict[str, str] = {}
    for r in sorted(glob.glob(f"{root}/**/*_rigorous.json", recursive=True)):
        doc = _load(r)
        if isinstance(doc, dict) and doc.get("benchmark") is not None:
            rig_by_bench.setdefault(str(doc["benchmark"]), r)
    out: list[tuple[str, str]] = []
    for m in matrices:
        doc = _load(m)
        if not isinstance(doc, dict):
            continue
        rig = rig_by_bench.get(str(doc.get("benchmark")))
        if rig:
            out.append((m, rig))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile oracle matrices against rigorous evals.")
    ap.add_argument("--root", default=str(_REPO / "experiments" / "final"),
                    help="glob <root>/**/oracle_matrix_*.json and pair by benchmark")
    ap.add_argument("--matrix", default=None, help="one explicit oracle_matrix_<bench>.json")
    ap.add_argument("--rigorous", default=None, help="its matching <bench>_rigorous.json")
    ap.add_argument("--report", action="store_true", help="print the report(s); always exit 0")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    ap.add_argument("--strict", action="store_true", help="fail (exit 1) on SUSPECT too, not only VOID")
    ap.add_argument("--z-ok", type=float, default=2.0, dest="z_ok", help="|z| tolerance to reconcile")
    ap.add_argument("--z-void", type=float, default=3.0, dest="z_void", help="|z| beyond which is VOID")
    args = ap.parse_args()

    pairs = _pairs(args.root, args.matrix, args.rigorous)
    if not pairs:
        print(f"no matrix/rigorous pairs found under {args.root} "
              "(pass --matrix and --rigorous explicitly)")
        sys.exit(2)

    summaries = []
    for matrix_path, rigorous_path in pairs:
        matrix, rigorous = _load(matrix_path), _load(rigorous_path)
        if matrix is None or rigorous is None:
            print(f"ERROR: could not read {matrix_path} / {rigorous_path}")
            sys.exit(2)
        s = reconcile(matrix, rigorous, z_ok=args.z_ok, z_void=args.z_void)
        summaries.append(s)
        print(render(s))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([s.to_dict() for s in summaries], indent=2))

    if args.report:
        return
    failed = [s for s in summaries if s.verdict == VOID or (args.strict and s.verdict == SUSPECT)]
    if failed:
        names = ", ".join(f"{s.benchmark}={s.verdict}" for s in failed)
        print(f"FAIL — {len(failed)} benchmark(s) did not reconcile: {names}")
        sys.exit(1)
    print(f"OK — {len(summaries)} benchmark(s) reconciled "
          f"({', '.join(s.benchmark for s in summaries)})")


if __name__ == "__main__":
    main()
