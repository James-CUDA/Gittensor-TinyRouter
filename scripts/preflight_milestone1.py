#!/usr/bin/env python3
"""Offline preflight for a Milestone-1 head submission (no GPU / no API).

Checks:
  1. weight sanity (shapes, finite, schema)
  2. param budget  — head (+ attention) < 1,000,000 params
  3. rate limit    — 1 submission per miner per day

Usage::

    python scripts/preflight_milestone1.py \\
        --submission submissions/alice/m1 \\
        --miner-name alice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.m1.gates import run_m1_gates  # noqa: E402
from trinity.m1.pack import load_milestone1_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--miner-name", required=True)
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument(
        "--skip-rate-limit",
        action="store_true",
        help="Skip 1/day check (local dry-run)",
    )
    args = ap.parse_args()

    pack = load_milestone1_pack(args.submission)
    results = run_m1_gates(
        pack,
        miner=args.miner_name,
        repo_root=args.repo_root,
        skip_rate_limit=args.skip_rate_limit,
    )
    print(f"[m1-preflight] {args.submission}  miner={args.miner_name}  n_params={pack.n_params:,}")
    failed = False
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        detail = f" — {r.reason}" if r.reason else ""
        print(f"  [{status}] {r.gate}{detail}")
        failed = failed or r.failed
    if failed:
        print("[m1-preflight] REJECTED", file=sys.stderr)
        sys.exit(1)
    print("[m1-preflight] OK — ready to eval / submit")
    sys.exit(0)


if __name__ == "__main__":
    main()
