#!/usr/bin/env python3
"""Print the overall cross-benchmark competition standings from leaderboard.json.

``trinity.leaderboard`` / ``trinity.submission.leaderboard`` report the frontier one
benchmark at a time; neither answers "who is winning overall?". This ranks miners by their
equal-weighted best-merged score across all benchmarks (each benchmark counts once; a
benchmark a miner never won counts as 0), so an overall leader must be strong everywhere.

    python scripts/standings_report.py
    python scripts/standings_report.py --file leaderboard.json --json standings.json

Read-only, pure/offline (no torch, no network). Degrades to "no miners yet" on the seed
leaderboard, so it is safe to run before the competition is populated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.standings import load_standings, render  # noqa: E402  (needs the sys.path insert)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overall cross-benchmark competition standings.")
    ap.add_argument("--file", default=str(_REPO / "leaderboard.json"), help="path to leaderboard.json")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    standings = load_standings(args.file)
    print(render(standings))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(standings.to_dict(), indent=2))


if __name__ == "__main__":
    main()
