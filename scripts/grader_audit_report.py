#!/usr/bin/env python3
"""Report an estimated grader error rate per benchmark (ORACLE_CEILING_DIAGNOSTIC §5 guard #2).

Reads built hidden-benchmark item file(s) — a JSON list of items, or a ``{"items": [...]}``
wrapper — pulls each item's ``benchmark`` + gold ``reference`` through the canonical
``adapters.hidden_item.from_protocol_item`` accessor (so this never drifts from the frozen
protocol's field names), and drives the FIXED grader over three probes per reference:
self-consistency, semantics-preserving fragility, and a clearly-wrong false-positive. It
prints the per-benchmark false-negative / false-positive / estimated-error rates and the
boundary cases to hand-review — the one oracle integrity guard that had no implementation.

    python scripts/grader_audit_report.py math500_items.json mmlu_items.json
    python scripts/grader_audit_report.py --root experiments --json grader_audit.json

Pure/offline: reads on-disk JSON only (no torch, no network, no GPU, no model calls).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from trinity.adapters.hidden_item import from_protocol_item
from trinity.analysis.grader_audit import audit, render


def _item_files(paths: list[str], root: str | None) -> list[str]:
    files = list(paths)
    if root:
        files += sorted(glob.glob(f"{root}/**/*items*.json", recursive=True))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _load_items(raw: Any) -> list[dict]:
    """Accept a bare list of items or a ``{"items": [...]}`` wrapper."""
    if isinstance(raw, dict):
        raw = raw.get("items", [])
    return [it for it in raw if isinstance(it, dict)] if isinstance(raw, list) else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-benchmark grader-error-rate audit.")
    ap.add_argument("files", nargs="*", help="hidden-benchmark item JSON file(s)")
    ap.add_argument("--root", default=None, help="also glob <root>/**/*items*.json")
    ap.add_argument("--sample", type=int, default=30, help="max boundary cases kept per benchmark")
    ap.add_argument("--seed", type=int, default=0, help="seed for boundary-sample selection")
    ap.add_argument("--json", default=None, dest="json_out", help="also write a JSON report")
    args = ap.parse_args()

    files = _item_files(args.files, args.root)
    if not files:
        print("no item JSONs given (pass files or --root)")
        return

    pairs: list[tuple[str, object]] = []
    for f in files:
        try:
            raw = json.loads(Path(f).read_text())
        except Exception:
            continue
        for item in _load_items(raw):
            try:
                canonical = from_protocol_item(item)
            except Exception:
                continue
            benchmark = str(canonical.get("benchmark") or "")
            if benchmark:
                pairs.append((benchmark, canonical.get("reference")))

    if not pairs:
        print("no (benchmark, reference) items found in the given files")
        return

    audits = audit(pairs, sample_size=args.sample, seed=args.seed)
    print(render(audits))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([a.to_dict() for a in audits], indent=2))


if __name__ == "__main__":
    main()
