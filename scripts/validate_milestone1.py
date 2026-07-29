#!/usr/bin/env python3
"""Validator: score M1 king vs challenger on the same held-out data.

Host workflow (mirrors competition ``pr_eval`` for Milestone 1)::

  1. Load fixed test split (pick data)
  2. Score current **king** pack (if any)
  3. Score **challenger** pack (miner submission)
  4. If challenger.composite >= king.composite + WIN_MARGIN (0.02)
     → MERGE (promote king); else REJECT

Usage::

    python scripts/validate_milestone1.py \\
        --challenger submissions/bob/m1 \\
        --miner-name bob \\
        --config 5-domain \\
        --features experiments/m1/test_features.npy \\
        --promote          # write new king into m1_leaderboard_*.json

    # First king (empty leaderboard): any gate-passing challenger wins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from trinity.m1.constants import DOMAIN_WEIGHT, WIN_MARGIN  # noqa: E402
from trinity.m1.gates import append_m1_attempt, run_m1_gates  # noqa: E402
from trinity.m1.leaderboard import (  # noqa: E402
    load_m1_leaderboard,
    promote_king,
    save_m1_leaderboard,
)
from trinity.m1.pack import load_milestone1_pack  # noqa: E402
from trinity.m1.scoring import compare_report, load_eval_split, score_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenger", type=Path, required=True, help="Challenger M1 pack dir")
    ap.add_argument("--miner-name", required=True)
    ap.add_argument("--config", default="5-domain", choices=("5-domain", "20-domain"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--hub", default="James-Cuda/tinyrouter-m1")
    ap.add_argument("--local", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Shared penultimate features (N,d_h) for fair king/challenger compare",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--win-margin", type=float, default=WIN_MARGIN)
    ap.add_argument("--domain-weight", type=float, default=DOMAIN_WEIGHT)
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument("--skip-rate-limit", action="store_true")
    ap.add_argument(
        "--promote",
        action="store_true",
        help="If challenger wins, write them as the new king on the leaderboard",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write verdict JSON")
    args = ap.parse_args()

    local = args.local or (
        args.repo_root / f"datasets/tinyrouter-m1/{args.config}/{args.split}.parquet"
    )
    ids, prompts, y_dom, y_diff = load_eval_split(
        args.config,
        args.split,
        hub=args.hub,
        local=local if local.exists() else None,
        limit=args.limit,
    )
    print(f"[data] config={args.config} split={args.split} n={len(prompts)}")

    feats = np.load(args.features) if args.features else None
    if feats is not None and feats.shape[0] != len(prompts):
        raise SystemExit(f"features N={feats.shape[0]} != data N={len(prompts)}")

    # --- gates on challenger ---
    chall_pack = load_milestone1_pack(args.challenger)
    gates = run_m1_gates(
        chall_pack,
        miner=args.miner_name,
        repo_root=args.repo_root,
        skip_rate_limit=args.skip_rate_limit,
    )
    for g in gates:
        print(f"[gate {'PASS' if g.ok else 'FAIL'}] {g.gate}: {g.reason}")
    if any(g.failed for g in gates):
        append_m1_attempt(
            args.repo_root,
            miner=args.miner_name,
            submission=str(args.challenger),
            ok=False,
        )
        raise SystemExit("challenger failed M1 gates — REJECT")

    # --- score king ---
    lb = load_m1_leaderboard(args.repo_root, args.config)
    king_metrics = None
    king_meta = None
    if lb.king is not None:
        king_path = Path(lb.king.submission)
        if not king_path.is_absolute():
            king_path = args.repo_root / king_path
        if not king_path.exists():
            print(f"[warn] king pack missing at {king_path}; treating as no king")
        else:
            print(f"[king] {lb.king.miner} @ {king_path} composite={lb.king.composite:.4f}")
            king_metrics = score_pack(
                king_path,
                prompts=prompts,
                y_domain=y_dom,
                y_diff=y_diff,
                features=feats,
                device=args.device,
                model_name=args.model_name,
                domain_weight=args.domain_weight,
            )
            king_meta = {"miner": lb.king.miner, "submission": str(king_path)}
            print(f"[king] rescored composite={king_metrics.composite:.4f}")
    else:
        print("[king] none — seat is open")

    # --- score challenger ---
    print(f"[challenger] {args.miner_name} @ {args.challenger}")
    chall_metrics = score_pack(
        chall_pack,
        prompts=prompts,
        y_domain=y_dom,
        y_diff=y_diff,
        features=feats,
        device=args.device,
        model_name=args.model_name,
        domain_weight=args.domain_weight,
    )
    print(f"[challenger] composite={chall_metrics.composite:.4f}")

    verdict = compare_report(
        king_metrics=king_metrics,
        challenger_metrics=chall_metrics,
        win_margin=args.win_margin,
        king_meta=king_meta,
        challenger_meta={
            "miner": args.miner_name,
            "submission": str(args.challenger),
        },
    )
    verdict["n"] = len(prompts)
    verdict["config"] = args.config
    verdict["split"] = args.split

    print(json.dumps(verdict, indent=2))

    append_m1_attempt(
        args.repo_root,
        miner=args.miner_name,
        submission=str(args.challenger),
        ok=bool(verdict["merge"]),
        metrics=chall_metrics.as_dict(),
    )

    if verdict["merge"] and args.promote:
        promote_king(
            lb,
            miner=args.miner_name,
            submission=str(args.challenger),
            composite=chall_metrics.composite,
            metrics=chall_metrics.as_dict(),
        )
        path = save_m1_leaderboard(args.repo_root, lb)
        print(f"[promote] new king → {path}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    if verdict["merge"]:
        print("[verdict] MERGE — challenger beats king")
        sys.exit(0)
    print("[verdict] REJECT — challenger did not beat king + margin", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
